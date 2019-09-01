#ifndef OUTPUT_HELPER_TEMPLATE_H
#define OUTPUT_HELPER_TEMPLATE_H

#include "output_helper.h"

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

namespace grendel
{

  template <int dim>
  void output(const CellProblem<dim> &cell_problem,
              const std::string &name)
  {
    const auto &discretization = cell_problem.discretization();
    const auto &dof_handler = cell_problem.dof_handler();
    const auto &solution = cell_problem.solution();

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    for(int k = 0; k < dim; ++k) {
      data_out.add_data_vector(solution[k], "xi_" + std::to_string(k));
    }

    data_out.build_patches(discretization.mapping());
    std::ofstream output(name + ".vtk");
    data_out.write_vtk(output);
  }

} /* namespace grendel */

#endif /* OUTPUT_HELPER_TEMPLATE_H */
