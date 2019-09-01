#include "simpleloop.h"

int main ()
{
  loki::SimpleLoop<2> simple_loop;

  /* If necessary, create empty parameter file and exit: */
  dealii::ParameterAcceptor::initialize("loki.prm");

  simple_loop.run();

  return 0;
}
