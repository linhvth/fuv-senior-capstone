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
from misc.data_generation import *
from misc.testing import testing_opt
from misc.functions import *

### Perform tests
if __name__ == '__main__':
  f, df = sphere_function()
  testing_opt(f, df, n_dims=3, true_global_min= np.array(([0, 0, 0])))

  # Test Himmelblau function (medium complexity non-convex)
  # f, df = modified_schwefel()
  # testing_opt(f, df, n_dims=1, init_point=None, step_size=1e-2, max_iter=1000, 
  #               max_stream=300, epsilon=1e-2, tolerance=1e-20)
  
  # Test with the 2-dim Rosenbrock function
  # f, df = rosenbrock_2d()
  # testing_opt(f, df, n_dims=2, init_point=np.array([4, -5]), step_size=1e-3)

  """Comment: both epsilon and tolerance affect to the solution obtain by SGD"""