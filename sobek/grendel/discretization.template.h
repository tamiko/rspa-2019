#ifndef DISCRETIZATION_TEMPLATE_H
#define DISCRETIZATION_TEMPLATE_H

#include "discretization.h"

#include "to_vector_function.template.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const std::string &subsection)
      : dealii::ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&Discretization::parse_parameters_callback, this));

    length_ = 0.7;
    add_parameter("length", length_, "Length of inner circle");

    radius_ = 1.0;
    add_parameter("radius", radius_, "Radius of inner circle");

    refinement_ = 5;
    add_parameter("initial refinement",
                  refinement_,
                  "Initial refinement of the geometry");

    order_mapping_ = 1;
    add_parameter("order mapping", order_mapping_, "Order of the mapping");

    order_finite_element_ = 1;
    add_parameter("order finite element",
                  order_finite_element_,
                  "Order of the finite element space");

    order_quadrature_ = 3;
    add_parameter(
        "order quadrature", order_quadrature_, "Order of the quadrature rule");
  }


  template <int dim>
  void Discretization<dim>::parse_parameters_callback()
  {
    /*
     * Create the Triangulation. We have $\Omega = [0,1]^dim$. Colorize
     * boundaries for periodic boundary conditions and mark material IDs
     * accordingly.
     */

    if (!triangulation_)
      triangulation_.reset(new Triangulation<dim>);

    auto &triangulation = *triangulation_;
    triangulation.clear();

    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

    triangulation.begin_active()->face(0)->set_boundary_id(0);
    triangulation.begin_active()->face(1)->set_boundary_id(2);
    triangulation.begin_active()->face(2)->set_boundary_id(1);
    triangulation.begin_active()->face(3)->set_boundary_id(3);

    triangulation.refine_global(refinement_);

    for (const auto &it : triangulation.active_cell_iterators()) {
      const auto cell_center = it->center();
      if (cell_center[1] > 0.5)
        it->set_material_id(1);
    }

    /*
     * Create an FE_Q^dim displacement field for the unit cell and populate
     * a MappingQEulerian object with it.
     */

    mapping_fe_.reset(new FESystem<dim>(FE_Q<dim>(order_finite_element_), dim));
    mapping_dof_handler_.reset(new DoFHandler<dim>());
    mapping_dof_handler_->initialize(triangulation, *mapping_fe_);

    mapping_displacement_.reset(
        new Vector<double>(mapping_dof_handler_->n_dofs()));

    const auto displacement = [&](const Point<dim> point) -> Tensor<1, dim> {

      Tensor<1, dim> displacement;

      const double tmp = std::sin(M_PI * point[0]);
      const double scale = 0.4;
      displacement[1] = scale * tmp * tmp;

      return displacement;
    };

    VectorTools::interpolate(*mapping_dof_handler_,
                             to_vector_function<dim, double>(displacement),
                             *mapping_displacement_);

    mapping_.reset(new MappingQEulerian<dim>(
        order_mapping_, *mapping_dof_handler_, *mapping_displacement_));

    /*
     * Populate the rest:
     */

    finite_element_.reset(
        new FESystem<dim>(FE_Q<dim>(order_finite_element_), 2));

    finite_element_ho_.reset(
        new FESystem<dim>(FE_Q<dim>(order_finite_element_ + 1), 2));

    quadrature_.reset(new QGauss<dim>(order_quadrature_));
  }

} /* namespace grendel */

#endif /* DISCRETIZATION_TEMPLATE_H */
