#ifndef DISCRETIZATION_TEMPLATE_H
#define DISCRETIZATION_TEMPLATE_H

#include "discretization.h"

#include "to_vector_function.template.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&Discretization<dim>::parse_parameters_callback, this));

    geometry_scaling_ = 15.0;
    add_parameter("geometry scaling",
                  geometry_scaling_,
                  "Scaling parameter for unit square and unit ball");

    refinement_ = 5;
    add_parameter("initial refinement",
                  refinement_,
                  "Initial refinement of the geometry");

    layer_position_ = 2.5;
    add_parameter("layer position",
                  layer_position_,
                  "upper and lower height of the first and last layer");

    no_layers_ = 2;
    add_parameter("no layers",
                  no_layers_,
                  "Number of layers expressed as an exponent of 2, i.e. 2**i "
                  "layers will be created.");

    order_mapping_ = 1;
    add_parameter("order mapping",
                  order_mapping_,
                  "Order of the mapping");

    order_finite_element_ = 1;
    add_parameter("order finite element",
                  order_finite_element_,
                  "Order of the finite element space");

    order_quadrature_ = 3;
    add_parameter("order quadrature",
                  order_quadrature_,
                  "Order of the quadrature rule");

    displacement_period_ = 3.0;
    add_parameter("displacement period",
                  displacement_period_,
                  "Displacement field - period");

    displacement_amplitude_ = 0.5;
    add_parameter("displacement amplitude",
                  displacement_amplitude_,
                  "Displacement field - amplitude");
  }


  template <int dim>
  void Discretization<dim>::parse_parameters_callback()
  {
    /*
     * Create the Triangulation. Set material ids accordingly.
     */

    if (!triangulation_)
      triangulation_.reset(new Triangulation<dim>);

    auto &triangulation = *triangulation_;
    triangulation.clear();

    /*
     * Create triangulation, fix layer hight, and apply gobal and local
     * refinement:
     */

    dealii::GridGenerator::subdivided_hyper_cube(
        triangulation, 6, -geometry_scaling_, geometry_scaling_);

    for (auto it : triangulation.cell_iterators()) {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {
        auto &vertex = it->vertex(v);
        if (std::abs(std::abs(vertex[1]) - geometry_scaling_ / 3.0) < 1.e-10) {
          vertex[1] *= layer_position_ / (geometry_scaling_ / 3.0);
        }
      }
    }

    triangulation.refine_global(refinement_);

    if (refinement_ < no_layers_) {
      for (unsigned int i = 0; i < (no_layers_ - refinement_); ++i) {
        for (auto it : triangulation.cell_iterators()) {
          const auto center = it->center();
          if (std::abs(center[1]) < layer_position_)
            it->set_refine_flag();
        }
        triangulation.execute_coarsening_and_refinement();
      }
    }

    /*
     * And colorize:
     */

    for (const auto &it : triangulation.active_cell_iterators()) {
      const auto center = it->center();

      if (std::abs(center[1]) > layer_position_) {
        it->set_material_id(3);
      } else {
        if (std::sin(std::pow(2, no_layers_) * M_PI *
                     (center[1] / layer_position_ + 1.0)) > 0.)
          it->set_material_id(1);
          else
          it->set_material_id(2);
      }
    }

    /*
     * Create an FE_Q^dim displacement field for the unit cell and populate
     * a MappingQEulerian object with it.
     */

    mapping_fe_.reset(new FESystem<dim>(FE_Q<dim>(order_mapping_), dim));
    mapping_dof_handler_.reset(new DoFHandler<dim>());
    mapping_displacement_.reset(new Vector<double>());

    update_mapping();

    /*
     * Populate the rest:
     */

    finite_element_.reset(new FE_Nedelec<dim>(order_finite_element_));

    finite_element_he_.reset(new FE_Nedelec<dim>(order_finite_element_ + 1));

    finite_element_projected_.reset(new FESystem<dim - 1, dim>(
        FE_Q<dim - 1, dim>(order_finite_element_ == 0 ? 1
                                                      : order_finite_element_),
        dim));

    quadrature_.reset(new QGauss<dim>(order_quadrature_));
  }


  template <int dim>
  void Discretization<dim>::update_mapping()
  {
    mapping_dof_handler_->initialize(*triangulation_, *mapping_fe_);

    mapping_displacement_->reinit(mapping_dof_handler_->n_dofs());

    const auto displacement = [&](const Point<dim> point) -> Tensor<1, dim> {
      Tensor<1, dim> displacement;

      const double tmp =
          std::sin(2. * M_PI * point[0] / displacement_period_);
      const double scale = 1.;
      displacement[1] = scale * displacement_amplitude_ * tmp * tmp;

      return displacement;
    };

    VectorTools::interpolate(*mapping_dof_handler_,
                             to_vector_function<dim, double>(displacement),
                             *mapping_displacement_);

    mapping_.reset(new MappingQEulerian<dim>(
        order_mapping_, *mapping_dof_handler_, *mapping_displacement_));
  }

} /* namespace grendel */

#endif /* DISCRETIZATION_TEMPLATE_H */
