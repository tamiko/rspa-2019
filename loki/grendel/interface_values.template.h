#ifndef INTERFACE_VALUES_TEMPLATE_H
#define INTERFACE_VALUES_TEMPLATE_H

#include "interface_values.h"

#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <algorithm>


namespace grendel
{
  using namespace dealii;
  typedef std::complex<double> value_type;


  template <int dim>
  InterfaceValues<dim>::InterfaceValues(
      const grendel::MaxwellProblem<dim> &maxwell_problem)
      : maxwell_problem_(&maxwell_problem)
  {
  }


  template <int dim>
  void InterfaceValues<dim>::extract_interface()
  {
    clear();

    const auto &discretization = maxwell_problem_->discretization();
    const auto &triangulation = discretization.triangulation();
    const auto &finite_element = discretization.finite_element_projected();

    //
    // Extract interface, i.e., the face that will make up the dim -1
    // triangulations. We have to store the corresponding cell in order to
    // project to the interface
    //

    std::set<std::pair<typename dealii::Triangulation<dim>::cell_iterator,
                       typename dealii::Triangulation<dim>::face_iterator>>
        interface;

    for (const auto &cell : triangulation.active_cell_iterators())
      for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
           ++f) {
        const auto face = cell->face(f);

        const auto id = cell->material_id();
        const auto neighbor_id =
            (face->at_boundary()) ? 255 : cell->neighbor(f)->material_id();

        if (neighbor_id == 255 || id >= neighbor_id || neighbor_id >= 4)
          continue;

        interface.insert(std::make_pair(cell, face));
      }

    {
      std::vector<dealii::Point<dim>> vertices;
      std::vector<dealii::CellData<dim - 1>> cells;

      for (const auto &match : interface) {

        const auto &face = match.second;

        dealii::CellData<dim - 1> cell_data;
        for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_face;
             ++v) {
          cell_data.vertices[v] = vertices.size();
          vertices.push_back(face->vertex(v));
        }
        cells.push_back(cell_data);
      }

      interface_triangulation_.create_triangulation(
          vertices, cells, dealii::SubCellData());
    }

    //
    // Now set up the dof handler and project solution
    //

    dealii::MappingQ<dim -1, dim> mapping_q(1); // FIXME

    dof_handler_.initialize(interface_triangulation_, finite_element);
    solution_.reinit(dof_handler_.n_dofs());

    // Build a quadrature rule out of the support points of the given
    // finite element:
    dealii::Quadrature<dim - 1> quadrature(
        finite_element.get_unit_support_points()); // FIXME
    const unsigned int n_q_points = quadrature.size();

    dealii::FEFaceValues<dim> fe_face_values(discretization.mapping(),
                                             discretization.finite_element(),
                                             quadrature,
                                             update_values |
                                                 update_quadrature_points);


    dealii::FEValues<dim - 1, dim> fe_values(
        mapping_q, finite_element, quadrature, update_default);

    dealii::Vector<value_type> dof_values(fe_face_values.dofs_per_cell);

    const auto n_dofs_per_cell = discretization.finite_element().dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const auto n_interface_dofs_per_cell = finite_element.dofs_per_cell;
    std::vector<types::global_dof_index> local_interface_dof_indices(
        n_interface_dofs_per_cell);

    auto dof_interface_cell = dof_handler_.begin_active();
    for (const auto &match : interface) {
      const auto &cell = match.first;
      const auto &face = match.second;

      typename DoFHandler<dim>::cell_iterator dof_cell(
          &cell->get_triangulation(),
          cell->level(),
          cell->index(),
          &maxwell_problem_->dof_handler());

      dof_cell->get_dof_indices(local_dof_indices);

      // get the correct index
      unsigned int f = 0;
      for (; cell->face(f) != face; ++f)
        ;
      fe_face_values.reinit(dof_cell, f);

      FEValuesViews::Vector<dim> fe_view(fe_face_values, 0);

      fe_values.reinit(dof_interface_cell);

      dof_cell->get_dof_indices(local_dof_indices);
      dof_interface_cell->get_dof_indices(local_interface_dof_indices);

      for (unsigned int q_point = 0; q_point < n_q_points / dim;
           ++q_point) { // FIXME
        Tensor<1, dim, value_type> value;

        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          value += maxwell_problem_->solution()[local_dof_indices[i]] *
                   fe_view.value(i, q_point);

        for (unsigned int d = 0; d < dim; ++d) {
          const auto global_index = local_interface_dof_indices
              [finite_element.component_to_system_index(d, q_point)];
          solution_[global_index] = value[d];
        }
      } /*for q_point*/
      ++dof_interface_cell;
    }   /*for match*/
  }


  template <int dim>
  void InterfaceValues<dim>::clear()
  {
    dof_handler_.clear();
    interface_triangulation_.clear();
    solution_.reinit(0);
  }

} /* namespace grendel */

#endif /* INTERFACE_VALUES_TEMPLATE_H */
