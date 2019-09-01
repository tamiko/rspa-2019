#ifndef MAXWELL_PROBLEM_TEMPLATE_H
#define MAXWELL_PROBLEM_TEMPLATE_H

#include "helper.h"
#include "maxwell_problem.h"

#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/vector.templates.h> // FIXME

namespace grendel
{
  using namespace dealii;
  typedef std::complex<double> value_type;

  template <int dim>
  MaxwellProblem<dim>::MaxwellProblem()
  {
  }

  template <int dim>
  MaxwellProblem<dim>::MaxwellProblem(
      const grendel::Discretization<dim> &discretization,
      const grendel::Coefficients<dim> &coefficients,
      const grendel::PerfectlyMatchedLayer<dim> &perfectly_matched_layer)
      : discretization_(&discretization)
      , coefficients_(&coefficients)
      , perfectly_matched_layer_(&perfectly_matched_layer)
  {
  }


  template <int dim>
  void MaxwellProblem<dim>::setup_system()
  {
    dof_handler_.initialize(discretization_->triangulation(),
                            discretization_->finite_element());

    deallog << "MaxwellProblem<dim>::setup_system()" << std::endl;
    deallog << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler_);

    setup_constraints();

    DynamicSparsityPattern c_sparsity(dof_handler_.n_dofs(),
                                      dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(
        dof_handler_, c_sparsity, affine_constraints_, false);
    sparsity_pattern_.copy_from(c_sparsity);

    system_matrix_.reinit(sparsity_pattern_);
    weighted_mass_matrix_.reinit(sparsity_pattern_);

    system_right_hand_side_.reinit(dof_handler_.n_dofs());
    solution_.reinit(dof_handler_.n_dofs());

    dual_system_right_hand_side_.reinit(dof_handler_.n_dofs());
    dual_solution_.reinit(dof_handler_.n_dofs());
  }


  template <int dim>
  void MaxwellProblem<dim>::setup_constraints()
  {
    affine_constraints_.clear();

    DoFTools::make_hanging_node_constraints(dof_handler_, affine_constraints_);

    affine_constraints_.close();
  }


  // internal data for parallelized assembly
  namespace
  {
    template <int dim>
    class AssemblyScratchData
    {
    public:
      AssemblyScratchData(const AssemblyScratchData<dim> &assembly_scratch_data)
          : AssemblyScratchData(assembly_scratch_data.discretization_)
      {
      }

      AssemblyScratchData(const grendel::Discretization<dim> &discretization)
          : discretization_(discretization)
          , fe_values_(discretization_.mapping(),
                       discretization_.finite_element(),
                       discretization_.quadrature(),
                       update_values | update_gradients |
                           update_quadrature_points | update_JxW_values)
          , face_quadrature_(3) // FIXME
          , fe_face_values_(discretization_.mapping(),
                            discretization_.finite_element(),
                            face_quadrature_,
                            update_values | update_gradients |
                                update_quadrature_points |
                                update_normal_vectors | update_JxW_values)
      {
      }

      const grendel::Discretization<dim> &discretization_;
      FEValues<dim> fe_values_;
      const QGauss<dim - 1> face_quadrature_;
      FEFaceValues<dim> fe_face_values_;
    };

    template <int dim>
    class AssemblyCopyData
    {
    public:
      std::vector<types::global_dof_index> local_dof_indices_;
      FullMatrix<value_type> cell_matrix_;
      FullMatrix<value_type> cell_weighted_mass_matrix_;
      Vector<value_type> cell_rhs_;
    };

  } /* anonymous namespace */


  template <int dim>
  void MaxwellProblem<dim>::assemble_system()
  {
    deallog << "MaxwellProblem<dim>::assemble_system()" << std::endl;

    this->system_matrix_ = 0.;
    this->weighted_mass_matrix_ = 0.;
    this->system_right_hand_side_ = 0.;
    this->dual_system_right_hand_side_ = 0.;
    this->dual_solution_ = 0.;
    this->solution_ = 0.;

    const unsigned int dofs_per_cell =
        this->discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = this->discretization_->quadrature().size();

    /* The local, per-cell assembly routine: */
    const std::function<void(
        const typename DoFHandler<dim>::active_cell_iterator &,
        AssemblyScratchData<dim> &,
        AssemblyCopyData<dim> &)>
        local_assemble_system = [&](const typename DoFHandler<
                                        dim>::active_cell_iterator &cell,
                                    AssemblyScratchData<dim> &scratch,
                                    AssemblyCopyData<dim> &copy) {
          static const auto imag = std::complex<double>(0., 1.);

          auto &cell_matrix = copy.cell_matrix_;
          auto &cell_weighted_mass_matrix = copy.cell_weighted_mass_matrix_;
          auto &cell_rhs = copy.cell_rhs_;
          auto &fe_face_values = scratch.fe_face_values_;
          auto &fe_values = scratch.fe_values_;
          auto &local_dof_indices = copy.local_dof_indices_;

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_weighted_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_rhs.reinit(dofs_per_cell);
          local_dof_indices.resize(dofs_per_cell);

          cell->get_dof_indices(local_dof_indices);

          fe_values.reinit(cell);
          FEValuesViews::Vector<dim> fe_view(fe_values, 0);
          const auto &quadrature_points = fe_values.get_quadrature_points();
          const auto id = cell->material_id();

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const auto position = quadrature_points[q_point];

            // Those virtual function calls are not as expensive as the
            // contraction further down below...
            auto mu_inv = coefficients_->mu_inv(position, id);
            auto epsilon = coefficients_->epsilon(position, id);
            const auto j_a = coefficients_->j_a(position, id);

            // FIXME 3D
            epsilon = perfectly_matched_layer_->pml(position) * epsilon;
            mu_inv /= perfectly_matched_layer_->scalar_pml(position);

            // index i for ansatz space, index j for test space

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              cell_rhs(i) += fe_view.value(i, q_point) * imag * j_a *
                             fe_values.JxW(q_point);

              const auto mu_inv_curl = mu_inv * fe_view.curl(i, q_point);
              const auto epsilon_vec = epsilon * fe_view.value(i, q_point);

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const auto curl_curl = fe_view.curl(j, q_point) * mu_inv_curl;
                const auto vec_vec = fe_view.value(j, q_point) * epsilon_vec;
                cell_matrix(j, i) +=
                    (curl_curl - vec_vec) * fe_values.JxW(q_point);

              } /* for */
            }   /* for */
          }     /* for */

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            const auto face = cell->face(f);

            fe_face_values.reinit(cell, f);

            const unsigned int n_face_q_points =
                scratch.face_quadrature_.size();
            const auto &quadrature_points =
                fe_face_values.get_quadrature_points();

            auto neighbor_id =
                (face->at_boundary()) ? 255 : cell->neighbor(f)->material_id();

            /* Assemble jump condition over interface or boundary condition: */
            if (id != neighbor_id || face->at_boundary()) {

              FEValuesViews::Vector<dim> fe_view(fe_face_values, 0);

              for (unsigned int q_point = 0; q_point < n_face_q_points;
                   ++q_point) {

                const auto position = quadrature_points[q_point];

                auto interface_sigma =
                    coefficients_->interface_sigma(position, id, neighbor_id);
                auto w = coefficients_->w(position, id, neighbor_id);

                // FIXME 3D
                interface_sigma =
                    perfectly_matched_layer_->interface_pml(position) *
                    interface_sigma;

                const auto normal = fe_face_values.normal_vector(q_point);

                const double JxW = fe_face_values.JxW(q_point);

                // index i for ansatz space, index j for test space
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const auto vec_t =
                      tangential_part(fe_view.value(i, q_point), normal);

                  const auto sigma_vec_t_JxW = interface_sigma * vec_t * JxW;
                  const auto w_vec_t_JxW = 2.0 * w * vec_t * JxW;

                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const Tensor<1, dim, value_type> phi_t =
                        tangential_part(fe_view.value(j, q_point), normal);

                    cell_weighted_mass_matrix(j, i) += phi_t * w_vec_t_JxW;

                    if (face->at_boundary()) {
                      cell_matrix(j, i) -= imag * phi_t * vec_t * JxW;
                    } else {
                      cell_matrix(j, i) -= 0.5 * imag * phi_t * sigma_vec_t_JxW;
                    }
                  } /* for */
                }   /* for */
              }     /* for */
            }       /* if */
          }         /* for loop over faces*/
        };

    /* The local-to-global copy routine: */
    const std::function<void(const AssemblyCopyData<dim> &)>
        copy_local_to_global = [this](const AssemblyCopyData<dim> &copy) {
          auto &cell_matrix = copy.cell_matrix_;
          auto &cell_weighted_mass_matrix = copy.cell_weighted_mass_matrix_;
          auto &cell_rhs = copy.cell_rhs_;
          auto &local_dof_indices = copy.local_dof_indices_;

          affine_constraints_.distribute_local_to_global(
              cell_matrix,
              cell_rhs,
              local_dof_indices,
              system_matrix_,
              system_right_hand_side_);

          affine_constraints_.distribute_local_to_global(
              cell_weighted_mass_matrix,
              local_dof_indices,
              weighted_mass_matrix_);
        };

    /* And run a workstream to assemble the matrix: */

    WorkStream::run(dof_handler_.begin_active(),
                    dof_handler_.end(),
                    local_assemble_system,
                    copy_local_to_global,
                    AssemblyScratchData<dim>(*this->discretization_),
                    AssemblyCopyData<dim>());
  }


  template <int dim>
  void MaxwellProblem<dim>::solve()
  {
    affine_constraints_.set_zero(solution_);
    affine_constraints_.set_zero(dual_solution_);

    deallog << "MaxwellProblem<dim>::solve()" << std::endl;

    /* workaround: no complex-valued direct solver available */

    /* Transfer system matrix to real valued matrix */

    SparsityPattern sparsity_pattern;
    {
      DynamicSparsityPattern c_sparsity(2 * dof_handler_.n_dofs(),
                                        2 * dof_handler_.n_dofs());
      for (const auto it : sparsity_pattern_) {
        const auto row = it.row();
        const auto column = it.column();

        c_sparsity.add(2 * row + 0, 2 * column + 0);
        c_sparsity.add(2 * row + 1, 2 * column + 0);
        c_sparsity.add(2 * row + 0, 2 * column + 1);
        c_sparsity.add(2 * row + 1, 2 * column + 1);
      }
      sparsity_pattern.copy_from(c_sparsity);
    }

    SparseMatrix<double> system_matrix;
    system_matrix.reinit(sparsity_pattern);

    for (const auto it : system_matrix_) {
      const std::complex<double> &value = it.value();
      const auto row = it.row();
      const auto column = it.column();

      system_matrix.set(2 * row + 0, 2 * column + 0, value.real());
      system_matrix.set(2 * row + 1, 2 * column + 0, value.imag());
      system_matrix.set(2 * row + 0, 2 * column + 1, -value.imag());
      system_matrix.set(2 * row + 1, 2 * column + 1, value.real());
    }

    {
      /* Transfer primal right hand side and solve system */

      Vector<double> solution;
      Vector<double> system_right_hand_side;

      solution.reinit(2 * dof_handler_.n_dofs());
      system_right_hand_side.reinit(2 * dof_handler_.n_dofs());

      const auto n_dofs = dof_handler_.n_dofs();
      for (unsigned int i = 0; i < n_dofs; ++i) {
        system_right_hand_side[2 * i] = system_right_hand_side_[i].real();
        system_right_hand_side[2 * i + 1] = system_right_hand_side_[i].imag();
      }

      SparseDirectUMFPACK solver;
      solver.initialize(system_matrix);
      solver.vmult(solution, system_right_hand_side);

      static const auto imag = std::complex<double>(0., 1.);
      for (unsigned int i = 0; i < n_dofs; ++i) {
        solution_[i] = solution[2 * i] + imag * solution[2 * i + 1];
      }

      affine_constraints_.distribute(solution_);

      /* Transfer dual right hand side and solve dual system */

      weighted_mass_matrix_.vmult(dual_system_right_hand_side_, solution_);
      dual_system_right_hand_side_ *= 2.;
      affine_constraints_.set_zero(dual_system_right_hand_side_);

      Vector<double> dual_solution;
      Vector<double> dual_system_right_hand_side;

      dual_solution.reinit(2 * dof_handler_.n_dofs());
      dual_system_right_hand_side.reinit(2 * dof_handler_.n_dofs());

      for (unsigned int i = 0; i < n_dofs; ++i) {
        dual_system_right_hand_side[2 * i] =
            dual_system_right_hand_side_[i].real();
        dual_system_right_hand_side[2 * i + 1] =
            dual_system_right_hand_side_[i].imag();
      }

      solver.Tvmult(dual_solution, dual_system_right_hand_side);

      for (unsigned int i = 0; i < n_dofs; ++i) {
        dual_solution_[i] =
            dual_solution[2 * i] + imag * dual_solution[2 * i + 1];
      }

      affine_constraints_.distribute(dual_solution_);
    } /*workaround*/

    functional_value_ = dual_system_right_hand_side_ * solution_;
  }


  template <int dim>
  void MaxwellProblem<dim>::clear()
  {
    dof_handler_.clear();
    sparsity_pattern_.reinit(0, 0, 0);
    affine_constraints_.clear();

    system_matrix_.clear();

    system_right_hand_side_.reinit(0);
    solution_.reinit(0);

    dual_system_right_hand_side_.reinit(0);
    dual_solution_.reinit(0);
  }

} /* namespace grendel */

#endif /* MAXWELL_PROBLEM_TEMPLATE_H */
