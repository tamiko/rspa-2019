#ifndef INTEGRATE_TEMPLATE_H
#define INTEGRATE_TEMPLATE_H

#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_values.h>

#include "discretization.h"
#include "coefficients.h"

namespace grendel
{
  using namespace dealii;

  namespace
  {
    template <int dim>
    class IntegrationScratchData
    {
    public:
      IntegrationScratchData(
          const IntegrationScratchData<dim> &Integration_scratch_data)
          : IntegrationScratchData(Integration_scratch_data.discretization_)
      {
      }

      IntegrationScratchData(const grendel::Discretization<dim> &discretization)
          : discretization_(discretization)
          , fe_values_(discretization_.mapping(),
                       discretization_.finite_element(),
                       discretization_.quadrature(),
                       update_values | update_gradients |
                           update_quadrature_points | update_JxW_values)
      {
      }

      const grendel::Discretization<dim> &discretization_;
      FEValues<dim> fe_values_;
    };

    template <int dim>
    class IntegrationCopyData
    {
    public:
      IntegrationCopyData()
          : sum(0.)
      {
      }

      double sum;
    };

  } /* anonymous namespace */


  template <int dim>
  double norm_difference(const MaxwellProblem<dim> &maxwell_problem)
  {
    double result = 0;

    const auto &discretization = maxwell_problem.discretization();

    const unsigned int dofs_per_cell =
        discretization.finite_element().dofs_per_cell;
    const unsigned int n_q_points = discretization.quadrature().size();

    /* local integration: */
    const std::function<void(
        const typename DoFHandler<dim>::active_cell_iterator &,
        IntegrationScratchData<dim> &,
        IntegrationCopyData<dim> &)> local_integration =
        [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
            IntegrationScratchData<dim> &scratch,
            IntegrationCopyData<dim> &copy) {
          auto &fe_values = scratch.fe_values_;

          fe_values.reinit(cell);
          FEValuesViews::Vector<dim> fe_view(fe_values, 0);

          std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          const auto &quadrature_points = fe_values.get_quadrature_points();

          copy.sum = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const auto &e_reference =
                maxwell_problem.coefficients().e_reference;

            auto difference = e_reference(quadrature_points[q_point]);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              difference -= maxwell_problem.solution()[local_dof_indices[i]] *
                            fe_view.value(i, q_point);
            copy.sum += difference.norm_square() * fe_values.JxW(q_point);
          }
        };

    const std::function<void(const IntegrationCopyData<dim> &)> sum_up = [&](
        const IntegrationCopyData<dim> &copy) { result += copy.sum; };

    /* And run a workstream to compute the norm:*/

    FEValues<dim> fe_values(discretization.mapping(),
                            discretization.finite_element(),
                            discretization.quadrature(),
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    WorkStream::run(maxwell_problem.dof_handler().begin_active(),
                    maxwell_problem.dof_handler().end(),
                    local_integration,
                    sum_up,
                    IntegrationScratchData<dim>(discretization),
                    IntegrationCopyData<dim>());

    return std::sqrt(result);
  }

} /* namespace grendel */

#endif /* INTEGRATE_TEMPLATE_H */
