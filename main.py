"""
This is main program for Linh Vu's Senior Capstone project

Linh Vu (2024)
"""

### IMPORT LIBRARIES

import time             
import numpy as np      
import math
from numpy.linalg import norm

from SGD import *
from data_generation import *


### LogReg Test => move to test


### Test 1: Simple convex functions
def square_function():
  f = lambda x: (x-2)**2
  df = lambda x: 2*(x-2)
  return f, df

### Test 2: Non-convex functions
def rosenbrock(X):
  """
  Rosenbrock function: 
    f(x) = sum((a_i - x[i-1])^2) for i = 1 to n-1, a = [1, 1.2, ..., n]
  """
  return


### Test 3: Perceptron

### Perform tests
if __name__ == '__main__':
  f, df = square_function()
  optimizer = SGD()
  optimizer.optimize(f, df, n_dims=1, step_size=1e-2, max_iter=1000, 
                     max_stream=100, epsilon=1e-5, tolerance=1e-2)
  print(optimizer.get_last_points())
  print(optimizer.get_solution())

  """Comment: both epsilon and tolerance affect to the solution obtain by SGD"""