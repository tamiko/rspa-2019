#ifndef COEFFICIENTS_TEMPLATE_H
#define COEFFICIENTS_TEMPLATE_H

#include "coefficients.h"


namespace grendel
{
  using namespace dealii;

  template <int dim>
  Coefficients<dim>::Coefficients(const std::string &subsection)
      : ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&Coefficients::parse_parameters_callback, this));

    material1_epsilon_ = 1.0;
    add_parameter("material 1 epsilon",
                  material1_epsilon_,
                  "Relative permittivity for material id 1.");

    material2_epsilon_ = 1.0;
    add_parameter("material 2 epsilon",
                  material2_epsilon_,
                  "Relative permittivity for material id 2.");

    d_ = 1.0;
    add_parameter("d",
                  d_,
                  "Interlayer spacing for Drude model.");

    ef_ = 1.0;
    add_parameter("E_F",
                  ef_,
                  "Fermi level.");

    omega_ = 1.0;
    add_parameter("omega", omega_, "Angular frequency.");

    length_ = 0.7;
    add_parameter("length", length_, "Length of inner circle");

    radius_ = 1.0;
    add_parameter("radius", radius_, "Radius of inner circle");
  }


  template <int dim>
  void Coefficients<dim>::parse_parameters_callback()
  {
    epsilon = [=](const dealii::Point<dim> & /*point*/,
                  const dealii::types::material_id &id) -> rank2_type {
      rank2_type result;

      for (unsigned int i = 0; i < dim; ++i)
        result[i][i] = id == 1 ? material2_epsilon_ : material1_epsilon_;

      return result;
    };

    eta = [=](const dealii::Point<dim> &/*point*/,
              const dealii::types::material_id &id1,
              const dealii::types::material_id &id2) -> rank2_type {
      rank2_type result;

      // Drude model
      static constexpr auto imag = std::complex<double>(0., 1.);
      const auto eta = 82.9 * ef_ / d_ / omega_ / (omega_ + 0.02 * imag);

      if ((id1 == 0 && id2 == 1) || (id1 == 1 && id2 == 0))
        for (unsigned int i = 0; i < dim; ++i)
          result[i][i] = eta;

      return result;
    };
  }

} /* namespace grendel */

#endif /* COEFFICIENTS_TEMPLATE_H */
