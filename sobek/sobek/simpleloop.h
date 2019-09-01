#ifndef SIMPLELOOP_H
#define SIMPLELOOP_H

#include <cell_problem.h>
#include <coefficients.h>
#include <discretization.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_acceptor.h>

namespace sobek
{
  /**
   * So...
   * This class is essentially a very, very big loop of spaghetti code ;-)
   */

  template <int dim>
  class SimpleLoop : public dealii::ParameterAcceptor
  {
  public:
    static_assert(dim == 2 || dim == 3, "Only supports dim == 2, or dim == 3");

    SimpleLoop();
    virtual ~SimpleLoop() final = default;

    virtual void run();

  private:
    /* Data: */

    std::string base_name_;

    unsigned int initial_resolution_;
    unsigned int no_cycles_;

    double omega_min_;
    double omega_max_;

    grendel::Discretization<dim> discretization;
    grendel::Coefficients<dim> coefficients;
    grendel::CellProblem<dim> cell_problem;

    std::map<double, std::array<std::complex<double>, dim * dim>> result;
  };

} /* namespace sobek */

#endif /* SIMPLELOOP_H */
