#ifndef COEFFICIENTS_H
#define COEFFICIENTS_H

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <complex>
#include <functional>

namespace grendel
{
  template <int dim>
  class Coefficients : public dealii::ParameterAcceptor
  {
  public:
    static_assert(dim == 2 || dim == 3, "Only supports dim == 2, or dim == 3");

    typedef std::complex<double> rank0_type;
    typedef dealii::Tensor<1, dim, rank0_type> rank1_type;
    typedef dealii::Tensor<2, dim, rank0_type> rank2_type;

    Coefficients(const std::string &subsection = "Coefficients");
    virtual ~Coefficients() final = default;

    void parse_parameters_callback();

    /*
     * Material parameters
     */

    std::function<rank2_type(const dealii::Point<dim> &,
                             const dealii::types::material_id &)>
        epsilon;

    std::function<rank2_type(const dealii::Point<dim> &,
                             const dealii::types::material_id &,
                             const dealii::types::material_id &)>
        eta;

  private:
    double length_;
    double radius_;

    std::complex<double> material1_epsilon_;
    std::complex<double> material2_epsilon_;

    std::complex<double> d_;
    std::complex<double> ef_;
    std::complex<double> omega_;
  };

} /* namespace grendel */

#endif /* COEFFICIENTS_H */
