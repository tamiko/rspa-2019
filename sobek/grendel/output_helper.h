#ifndef OUTPUT_HELPER_H
#define OUTPUT_HELPER_H

#include "cell_problem.h"

namespace grendel
{
  template <int dim>
  void output(const CellProblem<dim> &cell_problem, const std::string &name);

} /* namespace grendel */

#endif /* OUTPUT_HELPER_H */
