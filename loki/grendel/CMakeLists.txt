ADD_LIBRARY(grendel SHARED
  coefficients.cc
  discretization.cc
  error_indicators.cc
  interface_values.cc
  maxwell_problem.cc
  output_helper.cc
  perfectly_matched_layer.cc
  )
DEAL_II_SETUP_TARGET(grendel)
