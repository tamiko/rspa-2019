#ifndef INTERFACE_VALUES_H
#define INTERFACE_VALUES_H

#include "maxwell_problem.h"

#include <complex>

namespace grendel
{

  template <int dim>
  class InterfaceValues : public dealii::Subscriptor
  {
  public:
    typedef std::complex<double> value_type;

    InterfaceValues(const grendel::MaxwellProblem<dim> &maxwell_problem);

    /* Interface for computation: */

    virtual void run()
    {
      extract_interface();
    }

    virtual void extract_interface();

    virtual void clear();

  protected:
    dealii::SmartPointer<const grendel::MaxwellProblem<dim>> maxwell_problem_;

    dealii::Triangulation<dim - 1, dim> interface_triangulation_;
    ACCESSOR_READ_ONLY(interface_triangulation)

    dealii::DoFHandler<dim - 1, dim> dof_handler_;
    ACCESSOR_READ_ONLY(dof_handler)

    dealii::Vector<value_type> solution_;
    ACCESSOR_READ_ONLY(solution)
  };

} /* namespace grendel */

#endif /* INTERFACE_VALUES_H */
