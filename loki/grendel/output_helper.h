#ifndef OUTPUT_HELPER_H
#define OUTPUT_HELPER_H

#include "error_indicators.h"
#include "interface_values.h"
#include "maxwell_problem.h"

/*
 * This header file contains various helper functions to output the
 * functions and results stored in the Coefficients and MaxwellProblem
 * classes.
 */

namespace grendel
{
  template <int dim>
  void output(const MaxwellProblem<dim> &maxwell_problem,
              const ErrorIndicators<dim> &error_indicators,
              const std::string &name);

  template <int dim>
  void output(const InterfaceValues<dim> &interface_values,
              const std::string &name);

} /* namespace grendel */

#endif /* OUTPUT_HELPER_H */
