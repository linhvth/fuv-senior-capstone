"""
This is main program for Linh Vu's Senior Capstone project

Linh Vu (2024)
"""

### IMPORT LIBRARIES

import time             
import numpy as np      
import math
from numpy.linalg import norm

### Test 1: Simple convex functions
f = lambda x: x**2
df = lambda x: 2*x


### Test 2: Non-convex functions
def rosenbrock(X):
  """
  Rosenbrock function: 
    f(x) = sum((a_i - x[i-1])^2) for i = 1 to n-1, a = [1, 1.2, ..., n]
  """
  a = np.array([1] + [1.2 for _ in range(len(X) - 1)])
  return np.sum((a - X[:-1])**2) + 100 * np.sum((X[1:] - a**2)**2)


### Test 3: Perceptron