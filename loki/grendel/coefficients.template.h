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
        std::bind(&Coefficients<dim>::parse_parameters_callback, this));

    material1_epsilon_ = 1.;
    add_parameter("material 1 epsilon",
                  material1_epsilon_,
                  "Relative permittivity for material id 1.");

    material1_mu_inv_ = 1.;
    add_parameter("material 1 mu_inv",
                  material1_mu_inv_,
                  "Inverse, relative permeability for material id 1.");

    material2_epsilon_ = 1.;
    add_parameter("material 2 epsilon",
                  material2_epsilon_,
                  "Relative permittivity for material id 2.");

    material2_mu_inv_ = 1.;
    add_parameter("material 2 mu_inv",
                  material2_mu_inv_,
                  "Inverse, relative permeability for material id 2.");

    interface12_sigma_ = std::complex<double>{0.001, 0.2};
    add_parameter("interface 1 2 sigma",
                  interface12_sigma_,
                  "Rescaled surface conductivity between material id 1 and 2.");

    dipole_point_[1] = -2.75;
    add_parameter("j_a location",
                  dipole_point_,
                  "Location of the dipole (used for 'dipole' description)");

    dipole_radius_ = 0.15625;
    add_parameter("j_a radius",
                  dipole_radius_,
                  "Radius of the dipole (used for 'dipole' description)");

    dipole_value_[0] = 1.0;
    add_parameter("j_a value",
                  dipole_value_,
                  "Strength and orientiation of the dipole");
  }


  template <int dim>
  void Coefficients<dim>::parse_parameters_callback()
  {
    epsilon = [=](const dealii::Point<dim> & /*point*/,
                  const dealii::types::material_id &id) -> rank2_type {
      rank2_type result;
      for (unsigned int i = 0; i < dim; ++i)
        result[i][i] = id == 1 ? material1_epsilon_ : material2_epsilon_;
      return result;
    };

    mu_inv = [=](const dealii::Point<dim> & /*point*/,
                 const dealii::types::material_id &id) -> permeability_type {

      permeability_type result;
      for (unsigned int i = 0; i < dim; ++i)
        dealii::TensorAccessors::extract<dim == 2 ? 0 : 2>( //
            result,
            dealii::TableIndices<2>(i, i)) =
            id == 1 ? material1_mu_inv_ : material2_mu_inv_;
      return result;
    };

    interface_sigma = [=](const dealii::Point<dim> &/*point*/,
                          const dealii::types::material_id &id1,
                          const dealii::types::material_id &id2) -> rank2_type {
      rank2_type result;

      if (!((id1 == 1 && id2 == 2) || (id1 == 2 && id2 == 1)))
        return result;

      for (unsigned int i = 0; i < dim; ++i)
        result[i][i] = interface12_sigma_;

      return result;
    };

    w = [=](const dealii::Point<dim> &point,
            const dealii::types::material_id &id1,
            const dealii::types::material_id &id2) -> rank2_type {

      auto result = interface_sigma(point, id1, id2);
      for (unsigned int i = 0; i < dim; ++i) {
        auto &entry = result[i][i];
        entry = std::abs(entry) < 1.e-12 ? 0. : 1.;
      }

      return result;
    };

    j_a = [=](const dealii::Point<dim> &point,
              const dealii::types::material_id &id) -> rank1_type {
      const auto distance = (dipole_point_ - point).norm() / dipole_radius_;

      if (distance > 1.)
        return rank1_type();

      double scale =
          std::cos(distance * M_PI / 2.) * std::cos(distance * M_PI / 2.) /
          (M_PI / 2. - 2. / M_PI) / dipole_radius_ / dipole_radius_;

      if(id == 3)
        scale *= -1;

      return dipole_value_ * scale;
    };
  }

} /* namespace grendel */

#endif /* COEFFICIENTS_TEMPLATE_H */
