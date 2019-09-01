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
  /**
   * Coefficients for Time-Harmonic Maxwell's Equation
   *
   * This class is a wrapper around various rescaled tensor functions used
   * in the description of the time-harmonic Maxwell's Equations.
   */
  template <int dim>
  class Coefficients : public dealii::ParameterAcceptor
  {
  public:
    static_assert(dim == 2 || dim == 3, "Only supports dim == 2, or dim == 3");

    typedef std::complex<double> rank0_type;
    typedef dealii::Tensor<1, dim, rank0_type> rank1_type;
    typedef dealii::Tensor<2, dim, rank0_type> rank2_type;

    typedef dealii::Tensor<1, dim == 2 ? 1 : dim, rank0_type> curl_type;
    typedef dealii::Tensor<dim == 2 ? 0 : 2, dim, rank0_type> permeability_type;

    Coefficients(const std::string &subsection = "Coefficients");

    virtual void parse_parameters_callback() ;

    /*
     * Material parameters
     */

    std::function<rank2_type(const dealii::Point<dim> &,
                             const dealii::types::material_id &)>
        epsilon;

    std::function<permeability_type(const dealii::Point<dim> &,
                                    const dealii::types::material_id &)>
        mu_inv;

    std::function<rank2_type(const dealii::Point<dim> &,
                             const dealii::types::material_id &,
                             const dealii::types::material_id &)>
        interface_sigma;

    /*
     * Forcing for primal problem
     */

    std::function<rank1_type(const dealii::Point<dim> &,
                             const dealii::types::material_id &)>
        j_a;

    std::function<rank2_type(const dealii::Point<dim> &,
                             const dealii::types::material_id &,
                             const dealii::types::material_id &)>
        w;

  private:
    std::complex<double> material1_epsilon_;
    std::complex<double> material1_mu_inv_;

    std::complex<double> material2_epsilon_;
    std::complex<double> material2_mu_inv_;

    std::complex<double> interface12_sigma_;

    dealii::Point<dim> dipole_point_;
    double dipole_radius_;
    dealii::Tensor<1, dim> dipole_value_;
  };

} /* namespace grendel */

#endif /* COEFFICIENTS_H */
