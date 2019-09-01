#ifndef MAXWELL_PROBLEM_H
#define MAXWELL_PROBLEM_H

#include "helper.h"

#include "coefficients.h"
#include "discretization.h"
#include "perfectly_matched_layer.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>

#include <complex>

namespace grendel
{

  template <int dim>
  class MaxwellProblem : public dealii::ParameterAcceptor
  {
  public:
    typedef std::complex<double> value_type;

    MaxwellProblem();

    MaxwellProblem(
        const grendel::Discretization<dim> &discretization,
        const grendel::Coefficients<dim> &coefficients,
        const grendel::PerfectlyMatchedLayer<dim> &perfectly_matched_layer);

    /* Interface for computation: */

    virtual void run()
    {
      setup_system();
      assemble_system();
      solve();
    }

    virtual void setup_system();
    virtual void assemble_system();
    virtual void solve();

    virtual void clear();

  protected:
    virtual void setup_constraints();

    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    ACCESSOR_READ_ONLY(discretization)

    dealii::SmartPointer<const grendel::Coefficients<dim>> coefficients_;
    ACCESSOR_READ_ONLY(coefficients)

    dealii::SmartPointer<const grendel::PerfectlyMatchedLayer<dim>>
        perfectly_matched_layer_;
    ACCESSOR_READ_ONLY(perfectly_matched_layer)

    /* Implementation: */
    dealii::DoFHandler<dim> dof_handler_;
    ACCESSOR_READ_ONLY(dof_handler)

    dealii::SparsityPattern sparsity_pattern_;
    ACCESSOR_READ_ONLY(sparsity_pattern)

    dealii::AffineConstraints<value_type> affine_constraints_;
    ACCESSOR_READ_ONLY(affine_constraints)

    dealii::SparseMatrix<value_type> system_matrix_;
    ACCESSOR_READ_ONLY(system_matrix)

    dealii::SparseMatrix<value_type> weighted_mass_matrix_;
    ACCESSOR_READ_ONLY(weighted_mass_matrix)

    dealii::Vector<value_type> system_right_hand_side_;
    ACCESSOR_READ_ONLY(system_right_hand_side)

    dealii::Vector<value_type> solution_;
    ACCESSOR_READ_ONLY(solution)

    dealii::Vector<value_type> dual_system_right_hand_side_;
    ACCESSOR_READ_ONLY(dual_system_right_hand_side)

    dealii::Vector<value_type> dual_solution_;
    ACCESSOR_READ_ONLY(dual_solution)

    std::complex<double> functional_value_;
    ACCESSOR_READ_ONLY(functional_value)
  };

} /* namespace grendel */

#endif /* MAXWELL_PROBLEM_H */
