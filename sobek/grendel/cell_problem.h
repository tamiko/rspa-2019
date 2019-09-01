#ifndef CELL_PROBLEM_H
#define CELL_PROBLEM_H

#include "boilerplate.h"
#include "coefficients.h"
#include "discretization.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>

namespace grendel
{

  template <int dim>
  class CellProblem : public dealii::ParameterAcceptor
  {
  public:
    CellProblem(
        const grendel::Discretization<dim> &discretization,
        const grendel::Coefficients<dim> &coefficients,
        const std::string &subsection = "CellProblem");

    virtual ~CellProblem() final = default;

    /* Interface for computation: */

    virtual void run()
    {
      setup_system();
      assemble_system();
      solve();
      compute_functional();
    }

    virtual void setup_system();
    virtual void assemble_system();
    virtual void solve();
    virtual void compute_functional();

    virtual void clear();

  protected:
    virtual void setup_constraints();

    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    A_RO(discretization)

    dealii::SmartPointer<const grendel::Coefficients<dim>> coefficients_;
    A_RO(coefficients)

    /* Implementation: */

    dealii::DoFHandler<dim> dof_handler_;
    A_RO(dof_handler)

    dealii::SparsityPattern sparsity_pattern_;
    A_RO(sparsity_pattern)

    dealii::AffineConstraints<double> affine_constraints_;
    A_RO(affine_constraints)

    dealii::SparseMatrix<double> system_matrix_;
    A_RO(system_matrix)

    std::array<dealii::Vector<double>, dim> right_hand_side_;

    std::array<dealii::Vector<double>, dim> solution_;
    A_RO(solution)

    dealii::Tensor<2, dim, std::complex<double>> epsilon_effective_;
    A_RO(epsilon_effective)
  };

} /* namespace grendel */

#endif /* CELL_PROBLEM_H */
