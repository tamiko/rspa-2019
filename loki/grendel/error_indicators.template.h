#ifndef ERROR_INDICATORS_TEMPLATE_H
#define ERROR_INDICATORS_TEMPLATE_H

#include "error_indicators.h"
#include "helper.h"

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  ErrorIndicators<dim>::ErrorIndicators(
      const grendel::MaxwellProblem<dim> &maxwell_problem)
      : maxwell_problem_(&maxwell_problem)
  {
  }

  // internal data for parallelized assembly
  namespace
  {
    template <int dim>
    class AssemblyScratchData
    {
    public:
      AssemblyScratchData(const AssemblyScratchData<dim> &assembly_scratch_data)
          : AssemblyScratchData(assembly_scratch_data.discretization_,
                                assembly_scratch_data.finite_element_)
      {
      }

      AssemblyScratchData(const grendel::Discretization<dim> &discretization,
                          const dealii::FiniteElement<dim> &finite_element)
          : discretization_(discretization)
          , finite_element_(finite_element)
          , fe_values_he_(discretization_.mapping(),
                          finite_element_,
                          discretization_.quadrature(),
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values)
          , face_quadrature_(3) // FIXME
          , fe_face_values_he_(discretization_.mapping(),
                               finite_element_,
                               face_quadrature_,
                               update_values | update_gradients |
                                   update_quadrature_points |
                                   update_normal_vectors | update_JxW_values)
      {
      }

      const grendel::Discretization<dim> &discretization_;
      const dealii::FiniteElement<dim> &finite_element_;
      FEValues<dim> fe_values_he_;
      const QGauss<dim - 1> face_quadrature_;
      FEFaceValues<dim> fe_face_values_he_;
    };

    template <int dim>
    class AssemblyCopyData
    {
    public:
      unsigned int active_cell_index_;
      typename ErrorIndicators<dim>::value_type indicator_;
    };

  } /* anonymous namespace */

  template <int dim>
  void ErrorIndicators<dim>::compute_error_indicators()
  {
    const auto &discretization = maxwell_problem_->discretization();
    const auto &finite_element = discretization.finite_element();
    const auto &finite_element_he = discretization.finite_element_he();

    const auto &coefficients = maxwell_problem_->coefficients();

    const auto &perfectly_matched_layer =
        maxwell_problem_->perfectly_matched_layer();

    error_indicators_.reinit(discretization.triangulation().n_active_cells());

    const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
    const unsigned int dofs_per_cell_he = finite_element_he.dofs_per_cell;

    const unsigned int n_q_points = discretization.quadrature().size();

    /* Set up interpolation and restriction matrices: */

    std::array<FullMatrix<double>, GeometryInfo<dim>::max_children_per_cell>
        restriction_matrices;
    for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_cell;
         ++i) {
      const auto &prolongation = finite_element_he.get_prolongation_matrix(i);
      const auto &restriction = finite_element_he.get_restriction_matrix(i);
      restriction_matrices[i].reinit(dofs_per_cell_he, dofs_per_cell_he);
      prolongation.mmult(restriction_matrices[i], restriction);
    }

    FullMatrix<double> interpolation_matrix(dofs_per_cell_he, dofs_per_cell);

    // FIXME This is a horrible hack.
    {
      // Unfortunately, FE_Nedelec does not implement a usable
      // interpolation - we just want the interpolation from one element to
      // the other. In order to compute those, create the projection mass
      // matrix M_fe2_fe1 and the mass matrix M_fe1_fe1 and set the
      // interpolation to M_fe2_fe1 * M_fe1_fe1 - this is almost the
      // interpolation matrix except that we forgot to integrate correctly
      // over subdimensional parts *cough*. Therefore filter out everything
      // that is less than 0.9:
      FullMatrix<double> mass_matrix(dofs_per_cell, dofs_per_cell);
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const auto &p = discretization.quadrature().point(q_point);
        const auto w = discretization.quadrature().weight(q_point);

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int d = 0; d < dim; ++d)
              mass_matrix(i, j) +=
                  finite_element.shape_value_component(
                      i, p, d) *
                  finite_element.shape_value_component(
                      j, p, d) *
                  w;

          for (unsigned int i = 0; i < dofs_per_cell_he; ++i)
            for (unsigned int d = 0; d < dim; ++d)
              interpolation_matrix(i, j) +=
                  finite_element_he.shape_value_component(i, p, d) *
                  finite_element.shape_value_component(j, p, d) * w;
        } /* for */
      } /* for */
      FullMatrix<double> inverse(dofs_per_cell, dofs_per_cell);
      inverse.invert(mass_matrix);
      FullMatrix<double> temp(dofs_per_cell_he, dofs_per_cell);
      interpolation_matrix.mmult(temp, inverse);
      interpolation_matrix = temp;

      // Filter:
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        for (unsigned int i = 0; i < dofs_per_cell_he; ++i)
          if (std::abs(interpolation_matrix(i, j)) < .9)
            interpolation_matrix(i, j) = 0.;
    }
    // end of FIXME

    /* The local, per-cell assembly routine: */
    auto local_assemble_system = [&](
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        AssemblyScratchData<dim> &scratch,
        AssemblyCopyData<dim> &copy) {
      static const auto imag = std::complex<double>(0., 1.);

      auto &fe_face_values_he = scratch.fe_face_values_he_;
      auto &fe_values_he = scratch.fe_values_he_;
      auto &active_cell_index = copy.active_cell_index_;
      auto &indicator = copy.indicator_;

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      Vector<value_type> primal(dofs_per_cell_he);
      Vector<value_type> primal_proj(dofs_per_cell_he);
      Vector<value_type> dual(dofs_per_cell_he);
      Vector<value_type> dual_proj(dofs_per_cell_he);
      {
        // Determine child index of parent cell - we need this
        // information to select the correct restriction matrix.
        unsigned int child_index = 0;
        for (; child_index < GeometryInfo<dim>::max_children_per_cell;
             ++child_index)
          if (cell->parent()->child(child_index) == cell)
            break;

        Vector<value_type> temp1(dofs_per_cell);

        cell->get_dof_values(maxwell_problem_->solution(), temp1);
        interpolation_matrix.vmult(primal, temp1);
        restriction_matrices[child_index].vmult(primal_proj, primal);
        primal_proj -= primal;

        cell->get_dof_values(maxwell_problem_->dual_solution(), temp1);
        interpolation_matrix.vmult(dual, temp1);
        restriction_matrices[child_index].vmult(dual_proj, dual);
        dual_proj -= dual;
      }

      fe_values_he.reinit(
          static_cast<typename Triangulation<dim>::active_cell_iterator>(cell));
      FEValuesViews::Vector<dim> fe_view_he(fe_values_he, 0);
      const auto &quadrature_points = fe_values_he.get_quadrature_points();

      active_cell_index = cell->active_cell_index();
      indicator = 0.;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

        const auto position = quadrature_points[q_point];
        const auto id = cell->material_id();

        auto mu_inv = coefficients.mu_inv(position, id);
        auto epsilon = coefficients.epsilon(position, id);

        const auto j_a = coefficients.j_a(position, id);

        // FIXME: 3D..
        epsilon =
            perfectly_matched_layer.pml(quadrature_points[q_point]) * epsilon;
        mu_inv /=
            perfectly_matched_layer.scalar_pml(quadrature_points[q_point]);

        using vec_type = decltype(imag * fe_view_he.value(0, 0));
        using curl_type = decltype(imag * fe_view_he.curl(0, 0));

        vec_type u;
        curl_type u_curl;
        vec_type z;
        curl_type z_curl;

        vec_type u_proj;
        curl_type u_curl_proj;
        vec_type z_proj;
        curl_type z_curl_proj;

        for (unsigned int i = 0; i < dofs_per_cell_he; ++i) {
          u += primal[i] * fe_view_he.value(i, q_point);
          u_curl += primal[i] * fe_view_he.curl(i, q_point);
          z += dual[i] * fe_view_he.value(i, q_point);
          z_curl += dual[i] * fe_view_he.curl(i, q_point);

          u_proj += primal_proj[i] * fe_view_he.value(i, q_point);
          u_curl_proj += primal_proj[i] * fe_view_he.curl(i, q_point);
          z_proj += dual_proj[i] * fe_view_he.value(i, q_point);
          z_curl_proj += dual_proj[i] * fe_view_he.curl(i, q_point);
        }

        // clang-format off
        indicator +=
            0.5 * (
                z_curl_proj * mu_inv            * u_curl
              + z_curl      * conjugate(mu_inv) * u_curl_proj
              - z_proj * epsilon            * u
              - z      * conjugate(epsilon) * u_proj
              - z_proj * imag * j_a
            ) * fe_values_he.JxW(q_point);
        // clang-format on
      }

      auto id = cell->material_id();
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const auto face = cell->face(f);
        const unsigned int n_face_q_points = scratch.face_quadrature_.size();

        auto neighbor_id =
            (face->at_boundary()) ? 255 : cell->neighbor(f)->material_id();

        if (id == neighbor_id && !face->at_boundary())
          continue;

        fe_face_values_he.reinit(
            static_cast<typename Triangulation<dim>::active_cell_iterator>(
                cell),
            f);
        FEValuesViews::Vector<dim> fe_face_view_he(fe_face_values_he, 0);

        const auto &quadrature_points =
            fe_face_values_he.get_quadrature_points();

        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {

          const auto position = quadrature_points[q_point];

          auto interface_sigma =
              coefficients.interface_sigma(position, id, neighbor_id);

          const auto w = coefficients.w(position, id, neighbor_id);

          interface_sigma = perfectly_matched_layer.interface_pml(position) *
                            interface_sigma;

          using vec_type = decltype(imag * fe_face_view_he.value(0, 0));

          const vec_type normal = fe_face_values_he.normal_vector(q_point);

          vec_type u;
          vec_type z;
          vec_type u_proj;
          vec_type z_proj;

          for (unsigned int i = 0; i < dofs_per_cell_he; ++i) {
            u += primal[i] * fe_face_view_he.value(i, q_point);
            z += dual[i] * fe_face_view_he.value(i, q_point);
            u_proj += primal_proj[i] * fe_face_view_he.value(i, q_point);
            z_proj += dual_proj[i] * fe_face_view_he.value(i, q_point);
          }

          const auto u_t = tangential_part(u, normal);
          const auto z_t = tangential_part(z, normal);
          const auto u_proj_t = tangential_part(u_proj, normal);
          const auto z_proj_t = tangential_part(z_proj, normal);

          if (!face->at_boundary()) {

            indicator -= 0.25 * imag *
                         (z_t * interface_sigma * u_proj_t +
                          z_proj_t * interface_sigma * u_t) *
                         fe_face_values_he.JxW(q_point);

            indicator -= 0.25 * imag * 2. * z_t * w * u_proj_t *
                         fe_face_values_he.JxW(q_point);
          }
        }
      } /* for loop over faces*/

    };

    Vector<value_type> eta(discretization.triangulation().n_active_cells());

    auto copy_local_to_global =
        [&eta](const AssemblyCopyData<dim> &copy) mutable {
          eta[copy.active_cell_index_] = copy.indicator_;
        };

    WorkStream::run(maxwell_problem_->dof_handler().begin_active(),
                    maxwell_problem_->dof_handler().end(),
                    local_assemble_system,
                    copy_local_to_global,
                    AssemblyScratchData<dim>(discretization, finite_element_he),
                    AssemblyCopyData<dim>());

    /*
     * The problem with above approach is that it does not result in
     * balanced local indicators (we omit the strong residual and boundary
     * terms due to lazyness). To work around this issue smooth patch-wise:
     */
    for (auto it : discretization.triangulation().active_cell_iterators()) {
      const auto n_childs = GeometryInfo<dim>::max_children_per_cell;

      value_type average;
      unsigned int num = 0;

      for (unsigned int i = 0; i < n_childs; ++i) {
        const auto &candidate = it->parent()->child(i);
        if (!candidate->has_children()) {
          average += eta[candidate->active_cell_index()];
          num++;
        }
      }
      average /= num;

      for (unsigned int i = 0; i < n_childs; ++i) {
        const auto &candidate = it->parent()->child(i);
        if (!candidate->has_children())
          error_indicators_[candidate->active_cell_index()] = std::abs(average);
      }
    }
  }


  template <int dim>
  void ErrorIndicators<dim>::clear()
  {
    error_indicators_.reinit(0);
  }

} /* namespace grendel */

#endif /* ERROR_INDICATORS_TEMPLATE_H */
