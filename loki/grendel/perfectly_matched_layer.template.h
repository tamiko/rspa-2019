#ifndef PERFECTLY_MATCHED_LAYER_TEMPLATE_H
#define PERFECTLY_MATCHED_LAYER_TEMPLATE_H

#include "perfectly_matched_layer.h"

namespace grendel
{
  using namespace dealii;

  template <int dim>
  PerfectlyMatchedLayer<dim>::PerfectlyMatchedLayer(
      const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(std::bind(
        &PerfectlyMatchedLayer<dim>::parse_parameters_callback, this));

    inner_radius_ = 12.;
    add_parameter("inner radius", inner_radius_, "Inner radius of the PML");

    outer_radius_ = 15.;
    add_parameter("outer radius", outer_radius_, "Outer radius of the PML");

    strength_ = 2.;
    add_parameter("strength", strength_, "Strength of the PML");
  }


  template <int dim>
  void PerfectlyMatchedLayer<dim>::parse_parameters_callback()
  {
    pml = [=](const dealii::Point<dim> &p) -> rank2_type {
      constexpr auto imag = rank0_type(0., 1.);

      rank2_type result;

      const double radius = p.norm();

      if (radius >= inner_radius_) {
        // We are within the perfectly matched layer.

        // Finite Element Methods for Maxwell's Equations - Peter Monk
        // p.  379 - 385

        // $ \sigma = \sigma_0(\rho - \rho_i)^2/(\rho_o - \rho_i)^2
        const double sigma =
            strength_ * (radius - inner_radius_) * (radius - inner_radius_) /
            (outer_radius_ - inner_radius_) / (outer_radius_ - inner_radius_);

        const rank0_type d = 1. + imag * sigma;

        // $ \bar\sigma = \frac{\sigma_0}{3}(\rho - \rho_i)^3/
        //   \rho / (\rho_o - \rho_i)^2
        const double sigma_bar =
            strength_ / 3. * (radius - inner_radius_) *
            (radius - inner_radius_) * (radius - inner_radius_) / radius /
            (outer_radius_ - inner_radius_) / (outer_radius_ - inner_radius_);

        const rank0_type d_bar = 1. + imag * sigma_bar;

        // $ \bar d * \bar d / d $ in the direction that shall be stretched
        // to infinity, $ d$ in all other directions.
        rank0_type factor[dim];
        factor[0] = d_bar * d_bar / d;
        for (unsigned int i = 1; i < dim; ++i)
          factor[i] = d;

        // Combine above factors with an appropriate rotation matrix
        switch (dim) {
        case 2:
          result[0][0] = factor[0] * p[0] * p[0] + factor[1] * p[1] * p[1];
          result[1][1] = factor[0] * p[1] * p[1] + factor[1] * p[0] * p[0];
          result[1][0] = (factor[0] - factor[1]) * p[0] * p[1];
          result[0][1] = (factor[0] - factor[1]) * p[0] * p[1];
          result /= radius * radius;
          break;
        case 3:
          AssertThrow(false, dealii::ExcMessage("Implement me"));
          break;
        }

      } else {

        for (unsigned int i = 0; i < dim; ++i)
          result[i][i] = 1.;
      }

      return result;
    };

    interface_pml = [=](const dealii::Point<dim> &p) -> rank2_type {
      constexpr auto imag = rank0_type(0., 1.);

      rank2_type result;

      const double radius = p.norm();

      if (radius >= inner_radius_) {
        // $ \sigma = \sigma_0(\rho - \rho_i)^2/(\rho_o - \rho_i)^2
        const double sigma =
            strength_ * (radius - inner_radius_) * (radius - inner_radius_) /
            (outer_radius_ - inner_radius_) / (outer_radius_ - inner_radius_);

        const rank0_type d = 1. + imag * sigma;

        // $ \bar\sigma = \frac{\sigma_0}{3}(\rho - \rho_i)^3/
        //   \rho / (\rho_o - \rho_i)^2
        const double sigma_bar =
            strength_ / 3. * (radius - inner_radius_) *
            (radius - inner_radius_) * (radius - inner_radius_) / radius /
            (outer_radius_ - inner_radius_) / (outer_radius_ - inner_radius_);

        const rank0_type d_bar = 1. + imag * sigma_bar;

        // $ \bar d / d $ in the direction that shall be stretched
        // to infinity, $1$ in normal direction of the interface, $d / \bar
        // d$ in the last directions.
        rank0_type factor[dim];

        factor[0] = d_bar / d;
        factor[1] = 1;
        if (dim == 3)
          factor[2 % dim] = d / d_bar;

        // Combine above factors with an appropriate rotation matrix
        switch (dim) {
        case 2:
          result[0][0] = factor[0] * p[0] * p[0] + factor[1] * p[1] * p[1];
          result[1][1] = factor[0] * p[1] * p[1] + factor[1] * p[0] * p[0];
          result[1][0] = (factor[0] - factor[1]) * p[0] * p[1];
          result[0][1] = (factor[0] - factor[1]) * p[0] * p[1];
          result /= radius * radius;
          break;
        case 3:
          AssertThrow(false, dealii::ExcMessage("Implement me"));
          break;
        }
      } else {
        for (unsigned int i = 0; i < dim; ++i)
          result[i][i] = 1.;
      }

      return result;
    };

    scalar_pml = [=](const dealii::Point<dim> &p) -> rank0_type {
      constexpr auto imag = rank0_type(0., 1.);

      const double radius = p.norm();

      if (radius >= inner_radius_) {
        const double sigma =
            strength_ * (radius - inner_radius_) * (radius - inner_radius_) /
            (outer_radius_ - inner_radius_) / (outer_radius_ - inner_radius_);
        const rank0_type d = 1. + imag * sigma;

        return d;

      } else {

        return 1.;
      }
    };
  }

} /* namespace grendel */

#endif /* PERFECTLY_MATCHED_LAYER_TEMPLATE_H */
