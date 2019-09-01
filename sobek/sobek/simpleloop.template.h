#ifndef SIMPLELOOP_TEMPLATE_H
#define SIMPLELOOP_TEMPLATE_H

#include "simpleloop.h"

#include <output_helper.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/grid/grid_out.h>

#include <iomanip>

namespace sobek
{
  using namespace dealii;
  using namespace grendel;

  template <int dim>
  SimpleLoop<dim>::SimpleLoop()
      : ParameterAcceptor("A - Base Parameter")
      , discretization("C - Discretization")
      , coefficients("B - Coefficients")
      , cell_problem(discretization, coefficients, "D - CellProblem")
  {
    base_name_ = "test";
    add_parameter("basename", base_name_, "base name for all output files");

    initial_resolution_ = 100;
    add_parameter("initial resolution", initial_resolution_, "initial resolution");

    no_cycles_ = 1;
    add_parameter("number of cycles", no_cycles_, "number of adaption cycles");

    omega_min_ = 2;
    add_parameter("omega min", omega_min_, "minimal angular frequency");

    omega_max_ = 3;
    add_parameter("omega max", omega_max_, "maximal angular frequency");
  }


  template <int dim>
  void SimpleLoop<dim>::run()
  {
    /*
     * Prepare output:
     */

    deallog.pop();
    deallog.push(DEAL_II_GIT_SHORTREV "+" SOBEK_GIT_SHORTREV);
#ifdef DEBUG
    deallog.depth_console(5);
    deallog.push("DEBUG");
#else
    deallog.depth_console(4);
#endif
    deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
    deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

    deallog << "[Init] Reading parameters and allocating objects... "
            << std::flush;

    ParameterAcceptor::initialize("sobek.prm");

    deallog << "[ OK ]" << std::endl;

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

    deallog.push(base_name_);

    /* clang-format off */
    /* Print out some info about current library and program versions: */
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# sobek   version " << std::setw(8) << SOBEK_VERSION
            << "  -  " << SOBEK_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;
    /* clang-format on */

    /*
     * On to the hard work:
     */

    /* clang-format off */
    deallog << std::endl;
    deallog << "    #####################################################" << std::endl;
    deallog << "    #########                                  ##########" << std::endl;
    deallog << "    #########            cycle "
       << std::right << std::setw(4) << 1 << "            ##########" << std::endl;
    deallog << "    #########                                  ##########" << std::endl;
    deallog << "    #####################################################" << std::endl;
    deallog << std::endl;
    /* clang-format on */

    deallog << "        output primal triangulation" << std::endl;
    std::ofstream output(base_name_ + "-triangulation.inp");
    GridOut().write_ucd(discretization.triangulation(), output);

    const auto compute = [&](double omega) {
      deallog << "        setting omega = " << omega << std::endl;
      ParameterAcceptor::prm.enter_subsection("B - Coefficients");
      ParameterAcceptor::prm.set("omega", std::to_string(omega) + ", 0");
      ParameterAcceptor::prm.leave_subsection();

      deallog << "        compute primal problem" << std::endl;
      cell_problem.run();

      deallog << "        output primal solution" << std::endl;

      grendel::output<dim>(cell_problem,
                           base_name_ + "-solution-omega-" +
                               std::to_string(omega));

      deallog << "        effective epsilon =";
      {
        std::array<std::complex<double>, dim * dim> value;
        for (unsigned int i = 0; i < dim; ++i) {
          for (unsigned int j = 0; j < dim; ++j) {
            const auto it = cell_problem.epsilon_effective()[i][j];

            if (std::abs(it) > 1.e-10) {
              value[i * dim + j] = it;
              deallog << " " << it;
            } else {
              value[i * dim + j] = 0.;
              deallog << " (0.,0.)";
            }
          }
        }
        deallog << std::endl;
        result[omega] = value;
      }
    };

    for (double omega = omega_min_; omega <= omega_max_;
         omega += (omega_max_ - omega_min_) / initial_resolution_) {
      compute(omega);
    }


    for (unsigned int cycle = 2; cycle <= no_cycles_; ++cycle) {
      /* clang-format off */
      deallog << std::endl;
      deallog << "    #####################################################" << std::endl;
      deallog << "    #########                                  ##########" << std::endl;
      deallog << "    #########            cycle "
         << std::right << std::setw(4) << cycle << "            ##########" << std::endl;
      deallog << "    #########                                  ##########" << std::endl;
      deallog << "    #####################################################" << std::endl;
      deallog << std::endl;
      /* clang-format on */

      deallog << "    adapt..." << std::endl;

      std::vector<std::pair<double, double>> candidates;

      for (auto it = result.begin(); std::next(it) != result.end(); ++it) {
        const auto it2 = std::next(it);

        double slope = 0;
        for (unsigned int i = 0; i < dim * dim; ++i) {
          slope += std::pow(std::abs(it2->second[i] - it->second[i]), 2.);
        }
        slope = std::sqrt(slope);

        candidates.push_back({(it2->first + it->first) / 2., slope});
      }

      std::sort(candidates.begin(),
                candidates.end(),
                [](auto left, auto right) -> auto {
                  return left.second > right.second;
                });

      for (unsigned int i = 0; i < (initial_resolution_ / 4); ++i) {
        compute(candidates[i].first);
      }
    }

    /* clang-format off */
    deallog << std::endl;
    deallog << "    #####################################################" << std::endl;
    deallog << "    #########                                  ##########" << std::endl;
    deallog << "    #########             result               ##########" << std::endl;
    deallog << "    #########                                  ##########" << std::endl;
    deallog << "    #####################################################" << std::endl;
    deallog << std::endl;
    deallog << std::endl;
    /* clang-format on */

    for (const auto &it : result) {
      deallog << "####    " << it.first;
      for (const auto &it2 : it.second)
        deallog << " " << it2;
      deallog << std::endl;
    }

    deallog.pop();
    deallog.detach();
  }



} /* namespace sobek */

#endif /* SIMPLELOOP_TEMPLATE_H */
