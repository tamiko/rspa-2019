#ifndef CELL_PROBLEM_TEMPLATE_H
#define CELL_PROBLEM_TEMPLATE_H

#include "cell_problem.h"
#include "helper.template.h"

#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>

namespace grendel
{
  using namespace dealii;


  template <int dim>
  CellProblem<dim>::CellProblem(
      const grendel::Discretization<dim> &discretization,
      const grendel::Coefficients<dim> &coefficients,
      const std::string &subsection /*= "CellProblem"*/)
      : ParameterAcceptor(subsection)
      , discretization_(&discretization)
      , coefficients_(&coefficients)
  {
  }


  template <int dim>
  void CellProblem<dim>::setup_system()
  {
    deallog << "CellProblem<dim>::setup_system()" << std::endl;

    dof_handler_.initialize(discretization_->triangulation(),
                            discretization_->finite_element());

    deallog << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler_);

    setup_constraints();

    DynamicSparsityPattern c_sparsity(dof_handler_.n_dofs(),
                                      dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(
        dof_handler_, c_sparsity, affine_constraints_, false);
    sparsity_pattern_.copy_from(c_sparsity);

    system_matrix_.reinit(sparsity_pattern_);

    for (auto &solution : solution_)
      solution.reinit(dof_handler_.n_dofs());

    for (auto &right_hand_side : right_hand_side_)
      right_hand_side.reinit(dof_handler_.n_dofs());
  }


  template <int dim>
  void CellProblem<dim>::setup_constraints()
  {
    affine_constraints_.clear();

    for (int i = 0; i < dim; ++i)
      DoFTools::make_periodicity_constraints(dof_handler_,
                                             /*b_id1    */ i,
                                             /*b_id2    */ dim + i,
                                             /*direction*/ i,
                                             affine_constraints_);

    DoFTools::make_hanging_node_constraints(dof_handler_, affine_constraints_);

    pin_a_dof(this->dof_handler_, this->affine_constraints_);

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
      FullMatrix<double> cell_matrix_;
      std::array<Vector<double>, dim> cell_rhs_;
      Tensor<2, dim, std::complex<double>> epsilon_effective_;
    };

  } /* anonymous namespace */


  template <int dim>
  void CellProblem<dim>::assemble_system()
  {
    deallog << "CellProblem<dim>::assemble_system()" << std::endl;

    system_matrix_ = 0.;

    for (auto &solution : solution_)
      solution = 0.;

    for (auto &right_hand_side : right_hand_side_)
      right_hand_side = 0.;

    const unsigned int dofs_per_cell =
        this->discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = this->discretization_->quadrature().size();

    /* The local, per-cell assembly routine: */
    auto local_assemble_system =
        [&](const auto &cell, auto &scratch, auto &copy) {

          static constexpr auto imag = std::complex<double>(0., 1.);
          static constexpr auto unit = std::complex<double>(1., 0.);

          auto &cell_matrix = copy.cell_matrix_;
          auto &cell_rhs = copy.cell_rhs_;
          auto &fe_face_values = scratch.fe_face_values_;
          auto &fe_values = scratch.fe_values_;
          auto &local_dof_indices = copy.local_dof_indices_;

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          for (auto &rhs : cell_rhs)
            rhs.reinit(dofs_per_cell);

          fe_values.reinit(cell);
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          FEValuesViews::Scalar<dim> fe_real(fe_values, 0);
          FEValuesViews::Scalar<dim> fe_imag(fe_values, 1);

          const auto &quadrature_points = fe_values.get_quadrature_points();
          const auto id = cell->material_id();

          const auto rot_x = 1.;
          const auto rot_y = -imag;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const auto position = quadrature_points[q_point];

            auto epsilon = coefficients_->epsilon(position, id);

            const auto JxW = fe_values.JxW(q_point);

            // index j for ansatz space, index i for test space

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {

              const auto grad = rot_x * fe_real.gradient(i, q_point) +
                                rot_y * fe_imag.gradient(i, q_point);

              const auto grad_epsilon_JxW = grad * epsilon * JxW;

              for (int k = 0; k < dim; ++k) {
                const auto rhs = grad_epsilon_JxW[k];
                cell_rhs[k](i) -= rhs.real();
              }

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const auto grad = fe_real.gradient(j, q_point) +
                                  imag * fe_imag.gradient(j, q_point);

                cell_matrix(i, j) += (grad_epsilon_JxW * grad).real();
              } /* for j */
            }   /* for i */
          }     /* for q */

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            const auto face = cell->face(f);

            auto neighbor_id =
                (face->at_boundary()) ? id : cell->neighbor(f)->material_id();

            /* Assemble jump condition over interface or boundary condition: */
            if (id == neighbor_id)
              continue;

            fe_face_values.reinit(cell, f);

            const unsigned int n_face_q_points =
                scratch.face_quadrature_.size();
            const auto &quadrature_points =
                fe_face_values.get_quadrature_points();

            FEValuesViews::Scalar<dim> fe_real(fe_face_values, 0);
            FEValuesViews::Scalar<dim> fe_imag(fe_face_values, 1);

            for (unsigned int q_point = 0; q_point < n_face_q_points;
                 ++q_point) {

              const auto position = quadrature_points[q_point];

              auto eta = coefficients_->eta(position, id, neighbor_id);
              const auto normal = unit * fe_face_values.normal_vector(q_point);
              const auto JxW = fe_face_values.JxW(q_point);

              // index j for ansatz space, index i for test space

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {

                const auto grad = rot_x * fe_real.gradient(i, q_point) +
                                  rot_y * fe_imag.gradient(i, q_point);
                const auto grad_t = tangential_part(grad, normal);

                const auto grad_t_eta = grad_t * eta * JxW;

                for (int k = 0; k < dim; ++k) {
                  const auto rhs = grad_t_eta[k];
                  cell_rhs[k](i) += 0.5 * rhs.real();
                }

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                  const auto grad = fe_real.gradient(j, q_point) +
                                    imag * fe_imag.gradient(j, q_point);
                  const auto grad_t = tangential_part(grad, normal);

                  cell_matrix(i, j) -= 0.5 * (grad_t_eta * grad_t).real();
                } /* for j */
              }   /* for i */
            }     /* for q */
          }       /* for loop over faces*/
        };

    /* The local-to-global copy routine: */
    auto copy_local_to_global = [this](const auto &copy) {
      auto &cell_matrix = copy.cell_matrix_;
      auto &cell_rhs = copy.cell_rhs_;
      auto &local_dof_indices = copy.local_dof_indices_;

      affine_constraints_.distribute_local_to_global(
          cell_matrix, local_dof_indices, system_matrix_);

      for (int k = 0; k < dim; ++k) {
        affine_constraints_.distribute_local_to_global(
            cell_rhs[k], local_dof_indices, right_hand_side_[k]);
      }
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
  void CellProblem<dim>::solve()
  {
    deallog << "CellProblem<dim>::solve()" << std::endl;

    SparseDirectUMFPACK solver;
    solver.initialize(system_matrix_);

    for (int k = 0; k < dim; ++k) {
      affine_constraints_.set_zero(solution_[k]);
      affine_constraints_.set_zero(right_hand_side_[k]);
      solver.vmult(solution_[k], right_hand_side_[k]);
      affine_constraints_.distribute(solution_[k]);
    }
  }


  template <int dim>
  void CellProblem<dim>::compute_functional()
  {
    deallog << "CellProblem<dim>::assemble_system()" << std::endl;

    epsilon_effective_ = 0.;

    const unsigned int dofs_per_cell =
        this->discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = this->discretization_->quadrature().size();

    /* The local, per-cell integration routine: */
    auto local_integrate =
        [&](const auto &cell, auto &scratch, auto &copy) {
          static constexpr auto imag = std::complex<double>(0., 1.);
          static constexpr auto unit = std::complex<double>(1., 0.);

          auto &fe_face_values = scratch.fe_face_values_;
          auto &fe_values = scratch.fe_values_;
          auto &local_dof_indices = copy.local_dof_indices_;

          auto &epsilon_effective = copy.epsilon_effective_;
          copy.epsilon_effective_ = 0.;

          fe_values.reinit(cell);
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          FEValuesViews::Scalar<dim> fe_real(fe_values, 0);
          FEValuesViews::Scalar<dim> fe_imag(fe_values, 1);

          const auto rot_x = 1.;
          const auto rot_y = -imag;

          const auto &quadrature_points = fe_values.get_quadrature_points();
          const auto id = cell->material_id();

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const auto position = quadrature_points[q_point];

            const auto epsilon = coefficients_->epsilon(position, id);
            const auto JxW = fe_values.JxW(q_point);

            Tensor<2, dim, std::complex<double>> grad_chi;
            Tensor<2, dim, std::complex<double>> grad_chi_conj;

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const auto grad = fe_real.gradient(i, q_point) +
                                imag * fe_imag.gradient(i, q_point);
              const auto grad_conj = rot_x * fe_real.gradient(i, q_point) +
                                     rot_y * fe_imag.gradient(i, q_point);
              for (int k = 0; k < dim; ++k) {
                grad_chi[k] += grad * solution_[k](local_dof_indices[i]);
                grad_chi_conj[k] +=
                    grad_conj * solution_[k](local_dof_indices[i]);
              }
            }
            for (int k = 0; k < dim; ++k) {
              grad_chi[k][k] += 1.;
              grad_chi_conj[k][k] += 1.;
            }

            epsilon_effective += grad_chi_conj * epsilon * transpose(grad_chi) * JxW;
          }     /* for q */

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            const auto face = cell->face(f);

            auto neighbor_id =
                (face->at_boundary()) ? id : cell->neighbor(f)->material_id();

            /* Assemble jump condition over interface or boundary condition: */
            if (id == neighbor_id)
              continue;

            fe_face_values.reinit(cell, f);

            const unsigned int n_face_q_points =
                scratch.face_quadrature_.size();
            const auto &quadrature_points =
                fe_face_values.get_quadrature_points();

            FEValuesViews::Scalar<dim> fe_real(fe_face_values, 0);
            FEValuesViews::Scalar<dim> fe_imag(fe_face_values, 1);

            for (unsigned int q_point = 0; q_point < n_face_q_points;
                 ++q_point) {

              const auto position = quadrature_points[q_point];

              auto eta = coefficients_->eta(position, id, neighbor_id);
              const auto normal = unit * fe_face_values.normal_vector(q_point);
              const auto JxW = fe_face_values.JxW(q_point);

              Tensor<2, dim, std::complex<double>> grad_t_chi;
              Tensor<2, dim, std::complex<double>> grad_t_chi_conj;

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const auto grad = fe_real.gradient(i, q_point) +
                                  imag * fe_imag.gradient(i, q_point);
                const auto grad_conj = rot_x * fe_real.gradient(i, q_point) +
                                       rot_y * fe_imag.gradient(i, q_point);
                for (int k = 0; k < dim; ++k) {
                  grad_t_chi[k] += grad * solution_[k](local_dof_indices[i]);
                  grad_t_chi_conj[k] +=
                      grad_conj * solution_[k](local_dof_indices[i]);
                }
              }


              for (int k = 0; k < dim; ++k) {
                grad_t_chi[k][k] += 1.;
                grad_t_chi[k] = tangential_part(grad_t_chi[k], normal);
                grad_t_chi_conj[k][k] += 1.;
                grad_t_chi_conj[k] =
                    tangential_part(grad_t_chi_conj[k], normal);
              }

              epsilon_effective -=
                  0.5 * grad_t_chi_conj * eta * transpose(grad_t_chi) * JxW;
            }     /* for q */
          }       /* for loop over faces*/
        };

    /* The local-to-global copy routine: */
    auto sum_up = [this](const auto &copy) {
      const auto &epsilon_effective = copy.epsilon_effective_;
      epsilon_effective_ += epsilon_effective;
    };

    /* And run a workstream to assemble the matrix: */

    WorkStream::run(dof_handler_.begin_active(),
                    dof_handler_.end(),
                    local_integrate,
                    sum_up,
                    AssemblyScratchData<dim>(*this->discretization_),
                    AssemblyCopyData<dim>());

  }


  template <int dim>
  void CellProblem<dim>::clear()
  {
    dof_handler_.clear();
    sparsity_pattern_.reinit(0, 0, 0);
    affine_constraints_.clear();

    system_matrix_.clear();

    for (auto &solution : solution_)
      solution.reinit(0);

    for (auto &right_hand_side : right_hand_side_)
      right_hand_side.reinit(0);
  }

} /* namespace grendel */

#endif /* CELL_PROBLEM_TEMPLATE_H */
