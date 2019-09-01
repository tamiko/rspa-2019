#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include "helper.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/tria.h>


namespace grendel
{
  template <int dim>
  class Discretization : public dealii::ParameterAcceptor
  {
  public:
    Discretization(const std::string &subsection = "Discretization");

    virtual void parse_parameters_callback();

    void update_mapping();

  protected:

    double geometry_scaling_;
    unsigned int refinement_;

    double layer_position_;
    unsigned int no_layers_;

    unsigned int order_mapping_;
    double displacement_amplitude_;
    double displacement_period_;

    unsigned int order_finite_element_;
    unsigned int order_quadrature_;

    std::unique_ptr<dealii::Manifold<dim>> manifold_;
    std::unique_ptr<dealii::Triangulation<dim>> triangulation_;
    ACCESSOR_READ_ONLY(triangulation)

    std::unique_ptr<const dealii::FiniteElement<dim>> mapping_fe_;
    std::unique_ptr<dealii::DoFHandler<dim>> mapping_dof_handler_;
    std::unique_ptr<dealii::Vector<double>> mapping_displacement_;
    std::unique_ptr<const dealii::Mapping<dim>> mapping_;
    ACCESSOR_READ_ONLY(mapping)

    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_;
    ACCESSOR_READ_ONLY(finite_element)

    std::unique_ptr<const dealii::FiniteElement<dim>> finite_element_he_;
    ACCESSOR_READ_ONLY(finite_element_he)

    std::unique_ptr<const dealii::FiniteElement<dim - 1, dim>>
        finite_element_projected_;
    ACCESSOR_READ_ONLY(finite_element_projected)

    std::unique_ptr<const dealii::Quadrature<dim>> quadrature_;
    ACCESSOR_READ_ONLY(quadrature)
  };

} /* namespace grendel */

#endif /* DISCRETIZATION_H */
