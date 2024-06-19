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
from utils.data_generation import *
from utils.functions import *

if __name__ == '__main__':
 

  # Test Himmelblau function (medium complexity non-convex)
  # f, df = modified_schwefel()
  # testing_opt(f, df, n_dims=1, init_point=None, step_size=1e-2, max_iter=1000, 
  #               max_stream=300, epsilon=1e-2, tolerance=1e-20)
  
  # Test with the 2-dim Rosenbrock function
  # f, df = rosenbrock_2d()
  # testing_opt(f, df, n_dims=2, init_point=np.array([4, -5]), step_size=1e-3)

  """Comment: both epsilon and tolerance affect to the solution obtain by SGD"""