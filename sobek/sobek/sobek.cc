#include "simpleloop.h"

#include <fstream>

int main()
{
  sobek::SimpleLoop<DIM> simple_loop;

  /*
   * If necessary, create empty parameter file and exit:
   */

  const auto filename = "sobek.prm";
  if (!std::ifstream(filename)) {
    std::ofstream file(filename);
    dealii::ParameterAcceptor::prm.print_parameters(
        file, dealii::ParameterHandler::OutputStyle::Text);
    return 0;
  }

  simple_loop.run();

  return 0;
}
