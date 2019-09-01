#ifndef ERROR_INDICATORS_H
#define ERROR_INDICATORS_H

#include "helper.h"

#include "maxwell_problem.h"

#include <complex>

namespace grendel
{

  template <int dim>
  class ErrorIndicators : public dealii::Subscriptor
  {
  public:
    using value_type = typename MaxwellProblem<dim>::value_type;

    ErrorIndicators(const grendel::MaxwellProblem<dim> &maxwell_problem);

    /* Interface for computation: */

    virtual void run()
    {
      compute_error_indicators();
    }

    virtual void compute_error_indicators();

    virtual void clear();

  protected:
    dealii::SmartPointer<const grendel::MaxwellProblem<dim>> maxwell_problem_;
    ACCESSOR_READ_ONLY(maxwell_problem)

    dealii::Vector<double> error_indicators_;
    ACCESSOR_READ_ONLY(error_indicators)
  };

} /* namespace grendel */

#endif /* ERROR_INDICATORS_H */
