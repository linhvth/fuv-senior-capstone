"""
This is main program for Linh Vu's Senior Capstone project

Linh Vu (2024)
"""

### IMPORT LIBRARIES

import time             # control the execution time
import numpy as np      # LinAlg/ matrices manipulation
import autograd.numpy as au
import math
from autograd import grad, jacobian   # get hessian
from numpy.linalg import norm

from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt

def tanh(x):                 # Define a function
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

x = np.linspace(-7, 7, 200)

plt.plot(x, tanh(x),
          x, egrad(tanh)(x),                                     # first  derivative
          x, egrad(egrad(tanh))(x),                              # second derivative
          x, egrad(egrad(egrad(tanh)))(x),                       # third  derivative
          x, egrad(egrad(egrad(egrad(tanh))))(x),                # fourth derivative
          x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x),         # fifth  derivative
          x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x))  # sixth  derivative

plt.show()
print("hi")
def main():
    return