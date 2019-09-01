#ifndef HELPER_H
#define HELPER_H

#include <deal.II/base/tensor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

/***********
 *|---------\
 *|         |
 *|         O
 *|        -|-
 *|         |
 *|        / \
 *|________________________
 *|GRAVEYARD OF BAD IDEAS*/

namespace grendel
{
  typedef std::complex<double> value_type;

  dealii::Tensor<0, 2, value_type> inline conjugate(
      const dealii::Tensor<0, 2, value_type> &tensor)
  {
    return std::conj((value_type)tensor);
  }

  dealii::Tensor<2, 2, value_type> inline conjugate(
      const dealii::Tensor<2, 2, value_type> &tensor)
  {
    dealii::Tensor<2, 2, value_type> result;
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int j = 0; j < 2; ++j)
        result[i][j] = std::conj(tensor[i][j]);
    return result;
  }

  dealii::Tensor<2, 3, value_type> inline conjugate(
      const dealii::Tensor<2, 3, value_type> &tensor)
  {
    dealii::Tensor<2, 3, value_type> result;
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        result[i][j] = std::conj(tensor[i][j]);
    return result;
  }


  template <int dim, typename Number>
  dealii::Tensor<1, dim, Number>
  tangential_curl(const dealii::Tensor<1, dim, Number> &tensor,
                  const dealii::Tensor<1, dim, Number> &normal)
  {
    return cross_product_3d(normal, tensor);
  }

  template <int dim, typename Number>
  dealii::Tensor<1, dim, Number>
  tangential_curl(const dealii::Tensor<1, 1, Number> &cross,
                  const dealii::Tensor<1, dim, Number> &normal)
  {
    return cross_product_2d(normal) * cross[0];
  }

  template <int dim, typename Number>
  dealii::Tensor<1, dim, Number>
  tangential_part(const dealii::Tensor<1, dim, Number> &tensor,
                  const dealii::Tensor<1, dim, Number> &normal)
  {
    dealii::Tensor<1, dim, Number> result;
    switch (dim) {
    case 2:
      result[0] = normal[1] * (tensor[0] * normal[1] - tensor[1] * normal[0]);
      result[1] = -normal[0] * (tensor[0] * normal[1] - tensor[1] * normal[0]);
      break;
    case 3:
      result = cross_product_3d(cross_product_3d(normal, tensor), normal);
      break;
    }

    return result;
  }

} /* namespace grendel */

/*
 * A convenience macro that automatically writes out an accessor (or
 * getter) function:
 *
 *   const Foo& bar() const
 *   {
 *      return bar_;
 *   }
 *
 * or
 *
 *   const Foo& bar() const
 *   {
 *      return *bar_;
 *   }
 *
 * depending on whether bar_ can be dereferenced, or not.
 */

namespace
{
  template <typename T>
  class is_dereferenciable
  {
    template <typename C>
    static auto test(...) -> std::false_type;

    template <typename C>
    static auto test(C *) -> decltype(*std::declval<C>(), std::true_type());

  public:
    typedef decltype(test<T>(nullptr)) type;
    static constexpr auto value = type::value;
  };

  template <typename T, typename>
  auto dereference(T &t) -> decltype(dereference(*t)) &;

  template <
      typename T,
      typename = typename std::enable_if<!is_dereferenciable<T>::value>::type>
  const T &dereference(T &t)
  {
    return t;
  }

  template <
      typename T,
      typename = typename std::enable_if<is_dereferenciable<T>::value>::type>
  auto dereference(T &t) -> const decltype(*t) &
  {
    return *t;
  }
} /* anonymous namespace */

#define ACCESSOR_READ_ONLY(member)                                             \
public:                                                                        \
  decltype(dereference(member##_)) &member() const                             \
  {                                                                            \
    return dereference(member##_);                                             \
  }                                                                            \
                                                                               \
protected:

#endif /* HELPER_H */
