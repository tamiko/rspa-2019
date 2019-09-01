#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "boilerplate.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>

namespace grendel
{
  template <int dim>
  class Discretization : public dealii::ParameterAcceptor
  {
  public:
    static_assert(dim == 2 || dim == 3, "Only supports dim == 2, or dim == 3");

    Discretization(const std::string &subsection = "Discretization");
    virtual ~Discretization() final = default;

    void parse_parameters_callback();

  protected:

    double length_;
    double radius_;

    unsigned int refinement_;

    unsigned int order_finite_element_;
    unsigned int order_mapping_;
    unsigned int order_quadrature_;

    std::unique_ptr<dealii::Triangulation<dim>> triangulation_;
    A_RO(triangulation)

    std::unique_ptr<const dealii::FiniteElement<dim>> mapping_fe_;
    std::unique_ptr<dealii::DoFHandler<dim>> mapping_dof_handler_;
    std::unique_ptr<dealii::Vector<double>> mapping_displacement_;
    std::unique_ptr<const dealii::Mapping<dim>> mapping_;
    A_RO(mapping)

    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_;
    A_RO(finite_element)

    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_ho_;
    A_RO(finite_element_ho)

    std::unique_ptr<const dealii::Quadrature<dim>> quadrature_;
    A_RO(quadrature)
  };

} /* namespace grendel */

#endif /* DISCRETIZATION_H */
