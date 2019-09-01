#ifndef PERFECTLY_MATCHED_LAYER_H
#define PERFECTLY_MATCHED_LAYER_H

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <complex>
#include <functional>

namespace grendel
{

  /**
   * A perfectly matched layer
   */
  template <int dim>
  class PerfectlyMatchedLayer : public dealii::ParameterAcceptor
  {
    static_assert(dim == 2, "Only supports dim == 2");

  public:
    typedef std::complex<double> rank0_type;
    typedef dealii::Tensor<2, dim, rank0_type> rank2_type;

    PerfectlyMatchedLayer(
        const std::string &subsection = "Perfectly Matched Layer");

    void parse_parameters_callback();

    std::function<rank2_type(const dealii::Point<dim> &)> pml;
    std::function<rank2_type(const dealii::Point<dim> &)> interface_pml;
    std::function<rank0_type(const dealii::Point<dim> &)> scalar_pml;

  private:
    double inner_radius_;
    double outer_radius_;
    double strength_;
  };
} /* namespace grendel */

#endif /* PERFECTLY_MATCHED_LAYER_H */
