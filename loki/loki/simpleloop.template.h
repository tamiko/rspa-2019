#ifndef SIMPLELOOP_TEMPLATE_H
#define SIMPLELOOP_TEMPLATE_H

#include "simpleloop.h"

#include <helper.h>
#include <integrate.template.h>
#include <output_helper.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <iomanip>

namespace loki
{
  using namespace dealii;
  using namespace grendel;

  template <int dim>
  SimpleLoop<dim>::SimpleLoop()
      : ParameterAcceptor("A - Base Parameter")
      , run_no_(0)
      , cycle_(0)
      , coefficients("B - Coefficients")
      , discretization("C - Discretization")
      , perfectly_matched_layer("B - Perfectly Matched Layer")
      , maxwell_problem(discretization,
                        coefficients,
                        perfectly_matched_layer)
      , error_indicators(maxwell_problem)
      , interface_values(maxwell_problem)
  {
    base_name_ = "test";
    add_parameter("basename",
                  base_name_,
                  "base name for all output files");

    no_cycles_ = 3;
    add_parameter("number of cycles",
                  no_cycles_,
                  "number of cycles");

    refine_adaptive_ = true;
    add_parameter("refine adaptive",
                  refine_adaptive_,
                  "use adaptive, local refinement with DWR method");
  }


  template <int dim>
  void SimpleLoop<dim>::run()
  {
    maxwell_problem.clear();
    error_indicators.clear();
    convergence_table.clear();

    deallog.pop();

    deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
    deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

    deallog << "[Init] Reading parameters and allocating objects... "
            << std::flush;

    ParameterAcceptor::initialize("loki.prm");

    deallog << "[ OK ]" << std::endl;

    /* clang-format off */
    /* Print out some info about current library and program versions: */
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# loki    version " << std::setw(8) << LOKI_VERSION
            << "  -  " << LOKI_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;
    /* clang-format on */

    /* Print out parameters to a prm file: */
    {
      std::ofstream output(base_name_ + "-parameter.prm");
      ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);
    }

    /* Prepare deallog: */
    std::ofstream filestream(base_name_ + "-deallog.log");
    deallog.attach(filestream);

    /* Print out parameters to deallog as well: */
    deallog << "SimpleLoop<dim>::run()" << std::endl;
    ParameterAcceptor::prm.log_parameters(deallog);

    deallog.push(DEAL_II_GIT_SHORTREV "+" LOKI_GIT_SHORTREV);
    deallog.push(base_name_);
#ifdef DEBUG
    deallog.depth_console(3);
    deallog.depth_file(3);
    deallog.push("DEBUG");
#else
    deallog.depth_console(2);
    deallog.depth_file(2);
#endif

    /*
     * On to the hard work:
     */

    /* This is the honorable, monumental main loop: */
    for (cycle_ = 1; cycle_ <= no_cycles_; ++cycle_) {
      /* clang-format off */
      deallog << std::endl;
      deallog << "    #####################################################" << std::endl;
      deallog << "    #########                                  ##########" << std::endl;
      deallog << "    #########            cycle "
         << std::right << std::setw(4) << cycle_ << "            ##########" << std::endl;
      deallog << "    #########                                  ##########" << std::endl;
      deallog << "    #####################################################" << std::endl;
      /* clang-format on */

      compute_primal_problem();

      if (refine_adaptive_) {
        compute_dual_problem();
        compute_error_indicators();
      }

      update_convergence_table();

      output_results();

      if (cycle_ != no_cycles_)
        refine_mesh();

    } /* The end of the honorable, monumental main loop */

    deallog.pop();
    deallog.detach();
  }


  /*
   * The acutal implementation of all those functions in the mainloop:
   */


  template <int dim>
  void SimpleLoop<dim>::compute_primal_problem()
  {
    /* clang-format off */
    deallog << std::endl;
    deallog << "SimpleLoop<dim>::compute_primal_problem()" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << "        #####      compute primal problem      ######" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << std::endl;
    /* clang-format on */

    deallog << "        compute primal problem" << std::endl;
    maxwell_problem.run();
  }


  template <int dim>
  void SimpleLoop<dim>::compute_dual_problem()
  {
    /* clang-format off */
    deallog << std::endl;
    deallog << "SimpleLoop<dim>::compute_dual_problem()" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << "        #####       compute dual problem       ######" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << std::endl;
    /* clang-format on */

    deallog << "        noop - already computed with primal problem"
            << std::endl;
  }


  template <int dim>
  void SimpleLoop<dim>::compute_error_indicators()
  {
    /* clang-format off */
    deallog << std::endl;
    deallog << "SimpleLoop<dim>::compute_error_indicators()" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << "        #####     compute error indicators     ######" << std::endl;
    deallog << "        #############################################" << std::endl;
    /* clang-format on */

    error_indicators.compute_error_indicators();
  }


  template <int dim>
  void SimpleLoop<dim>::output_results()
  {
    deallog << std::endl;

    deallog << "SimpleLoop<dim>::output_results()" << std::endl;

    if (cycle_ == 1) {
      deallog << "        output primal triangulation" << std::endl;
      std::ofstream output(base_name_ + "-triangulation.inp");
      GridOut().write_ucd(discretization.triangulation(), output);
    }

    deallog << "        output primal solution" << std::endl;

    output<dim>(maxwell_problem,
                error_indicators,
                base_name_ + "-solution-" + std::to_string(cycle_));

    deallog << "        extract interface solution" << std::endl;

    interface_values.extract_interface();

    if (cycle_ == 1) {
      deallog << "        output interface triangulation" << std::endl;
      std::ofstream output(base_name_ + "-interface_triangulation.inp");
      GridOut().write_ucd(interface_values.interface_triangulation(), output);
    }

    output<dim>(interface_values,
                base_name_ + "-interface-solution-" + std::to_string(cycle_));
  }


  template <int dim>
  void SimpleLoop<dim>::update_convergence_table()
  {
    deallog << std::endl;
    deallog << "SimpleLoop<dim>::update_convergence_table()" << std::endl;

    convergence_table.add_value("#Cycle", cycle_);

    const auto &triangulation = discretization.triangulation();

    convergence_table.add_value("#Cells", triangulation.n_active_cells());
    convergence_table.add_value("#DoFs",
                                maxwell_problem.dof_handler().n_dofs());


    std::complex<double> functional_value = maxwell_problem.functional_value();

    convergence_table.add_value("Func Re", std::real(functional_value));
    convergence_table.add_value("Func Im", std::imag(functional_value));

    for (std::string it : {"Func Re", "Func Im"}) {
      convergence_table.evaluate_convergence_rates(
          it, ConvergenceTable::reduction_rate_log2);
      convergence_table.set_scientific(it, true);
      convergence_table.set_precision(it, 3);
    }

    std::ofstream output(base_name_ + "-convergence-table.txt");
    convergence_table.write_text(output);

    deallog.get_console() << std::endl;
    deallog.get_console() << std::endl;
    convergence_table.write_text(deallog.get_console());
    deallog.get_console() << std::endl;
    deallog.get_console() << std::endl;
  }


  template <int dim>
  void SimpleLoop<dim>::refine_mesh()
  {
    /* clang-format off */
    deallog << std::endl;
    deallog << "SimpleLoop<dim>::refine_mesh()" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << "        #####            refine mesh           ######" << std::endl;
    deallog << "        #############################################" << std::endl;
    deallog << std::endl;
    /* clang-format on */

    auto &triangulation = discretization.triangulation();

    if (!refine_adaptive_) {
      triangulation.refine_global(1);

    } else {

      GridRefinement::refine_and_coarsen_fixed_number(
          triangulation, error_indicators.error_indicators(), 0.33, 0.0);

      triangulation.execute_coarsening_and_refinement();
    }

    discretization.update_mapping();
  }

} /* namespace loki */

#endif /* SIMPLELOOP_TEMPLATE_H */
