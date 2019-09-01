#include "output_helper.template.h"

namespace grendel
{
  /* instantiations */

  template void output<2>(const CellProblem<2> &, const std::string &);
  template void output<3>(const CellProblem<3> &, const std::string &);

} /* namespace grendel */
