#ifndef SIMPLELOOP_H
#define SIMPLELOOP_H

#include <coefficients.h>
#include <discretization.h>
#include <error_indicators.h>
#include <interface_values.h>
#include <maxwell_problem.h>
#include <perfectly_matched_layer.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

namespace loki
{
  template <int dim>
  class SimpleLoop : public dealii::ParameterAcceptor
  {
  public:
    static_assert(dim == 2 || dim == 3, "Only supports dim == 2, or dim == 3");

    SimpleLoop();

    void run();

  private:

    /* Implementation: */

    void compute_primal_problem();
    void compute_dual_problem();
    void compute_error_indicators();
    void output_results();
    void update_convergence_table();
    void refine_mesh();

    /* Data: */

    unsigned int run_no_;
    unsigned int cycle_;

    std::string base_name_;
    unsigned int no_cycles_;
    bool refine_adaptive_;

    grendel::Coefficients<dim> coefficients;

    grendel::Discretization<dim> discretization;

    grendel::PerfectlyMatchedLayer<dim> perfectly_matched_layer;

    grendel::MaxwellProblem<dim> maxwell_problem;
    grendel::ErrorIndicators<dim> error_indicators;

    grendel::InterfaceValues<dim> interface_values;

    dealii::ConvergenceTable convergence_table;
  };

} /* namespace loki */

#endif /* SIMPLELOOP_H */
