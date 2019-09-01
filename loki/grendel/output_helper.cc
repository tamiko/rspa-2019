#include "output_helper.template.h"

namespace grendel
{
  template void output<2>(const MaxwellProblem<2> &,
                          const ErrorIndicators<2> &,
                          const std::string &);


  template void output<2>(const InterfaceValues<2> &,
                          const std::string &);
} /* namespace grendel */
