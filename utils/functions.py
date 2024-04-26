"""
This module provides functions for generating data for Stochastic Gradient 
Descent (SGD) testing.

These functions offer tools to create datasets with specific characteristics 
suitable for evaluating SGD algorithms.
"""
import math
import numpy as np

class FunctionInfo:
  def __init__(self, name, function, derivative, lipschitz_constant_df, n_dims, global_min=None):
    self.name = name
    self.function = function
    self.derivative = derivative
    self.lipschitz_constant_df = lipschitz_constant_df
    self.global_min = global_min  # Optional field for global minimum
    self.n_dims = n_dims

def quadratic_func(n_dims):
  # Function definitions
  f_quad = lambda x: np.sum(x**2) 
  df_quad = lambda x: 2*x
  lipschitz_df_quad = 2
  global_min_quad = np.zeros(n_dims)

  # return function information object
  return FunctionInfo("Quadratic", f_quad, df_quad, lipschitz_df_quad, n_dims, global_min_quad)

def multivar_logistic():
  f_logistic = lambda x: 1 / (1 + np.exp(-np.sum(x)))
  df_logistic = lambda x: -np.exp(-np.sum(x)) * np.exp(x) / (1 + np.exp(-np.sum(x)))**2
  lipschitz_g_logistic = 1

  return FunctionInfo("Multivariate Logistic", f_logistic, df_logistic, lipschitz_g_logistic)

