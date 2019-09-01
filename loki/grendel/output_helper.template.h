#ifndef OUTPUT_HELPER_TEMPLATE_H
#define OUTPUT_HELPER_TEMPLATE_H

#include "output_helper.h"

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <tuple>
#include <fstream>

namespace grendel
{

  template <int dim>
  void output(const MaxwellProblem<dim> &maxwell_problem,
              const ErrorIndicators<dim> &error_indicators,
              const std::string &name)
  {
    const auto &discretization = maxwell_problem.discretization();
    const auto &dof_handler = maxwell_problem.dof_handler();

    dealii::FESystem<dim> fe(dealii::FE_Q<dim>(1), dim);
    dealii::DoFHandler<dim> dof_handler2;
    dof_handler2.initialize(discretization.triangulation(), fe);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    dealii::Vector<double> f_real(dof_handler.n_dofs());
    dealii::Vector<double> f_imag(dof_handler.n_dofs());
    dealii::Vector<double> e_real(dof_handler.n_dofs());
    dealii::Vector<double> e_imag(dof_handler.n_dofs());
    dealii::Vector<double> z_real(dof_handler.n_dofs());
    dealii::Vector<double> z_imag(dof_handler.n_dofs());

    // begin FIXME
    // Fix the library so that we can use complex number types in
    // DataOut...
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
      f_real[i] = maxwell_problem.system_right_hand_side()[i].real();
      f_imag[i] = maxwell_problem.system_right_hand_side()[i].imag();

      const auto value = maxwell_problem.solution()[i];
      e_real[i] = value.real();
      e_imag[i] = value.imag();
      z_real[i] = maxwell_problem.dual_solution()[i].real();
      z_imag[i] = maxwell_problem.dual_solution()[i].imag();
    }
    // end FIXME

    data_out.add_data_vector(f_real, "f_real");
    data_out.add_data_vector(f_imag, "f_imag");

    data_out.add_data_vector(e_real, "e_real");
    data_out.add_data_vector(e_imag, "e_imag");
    data_out.add_data_vector(z_real, "z_real");
    data_out.add_data_vector(z_imag, "z_imag");

    if (error_indicators.error_indicators().size() != 0) {
      data_out.add_data_vector(error_indicators.error_indicators(), "eta");
    }

    data_out.build_patches(maxwell_problem.discretization().mapping());

    std::ofstream output(name + ".vtk");
    data_out.write_vtk(output);
  }


  template <int dim>
  void output(const InterfaceValues<dim> &interface_values,
              const std::string &name)
  {
    dealii::MappingQ<dim - 1, dim> mapping_q(1); // FIXME

    const auto &dof_handler = interface_values.dof_handler();

    dealii::DataOut<dim - 1, dealii::DoFHandler<dim - 1, dim>> data_out;
    data_out.attach_dof_handler(dof_handler);

    dealii::Vector<double> solution_real(dof_handler.n_dofs());
    dealii::Vector<double> solution_imag(dof_handler.n_dofs());

    // begin FIXME
    // Fix the library so that we can use complex number types in
    // DataOut...
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
      solution_real[i] = interface_values.solution()[i].real();
      solution_imag[i] = interface_values.solution()[i].imag();
    }
    // end FIXME

    data_out.add_data_vector(solution_real, "solution_real");
    data_out.add_data_vector(solution_imag, "solution_imag");

    data_out.build_patches(mapping_q);

    {
      std::ofstream output(name + ".vtk");
      data_out.write_vtk(output);
    }
    {
      std::ofstream output(name + ".gnuplot");
      data_out.write_gnuplot(output);
    }
  }


} /* namespace grendel */

#endif /* OUTPUT_HELPER_TEMPLATE_H */
