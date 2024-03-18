"""

"""

### IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt

def generate_data(function, n_samples, lower_bound, upper_bound):
  """
  Generates data from a given function.

  Args:
      function (callable): The function to generate data from.
      n_samples (int): The number of data points to generate.
      lower_bound (float): The lower bound of the input range.
      upper_bound (float): The upper bound of the input range.

  Returns:
      tuple: A tuple containing the generated input data (X) 
             and output data (y).
  """
  X = np.random.uniform(low=lower_bound, 
                        high=upper_bound, 
                        size=n_samples)
  y = function(X)

  return X, y

def visualize_data(X, y, function, lower_bound, upper_bound, title):
  """
  Visualizes the generated data and the function.

  Args:
      X (np.ndarray): The input data.
      y (np.ndarray): The output data.
      function (callable): The function used to generate the data.
      lower_bound (float): The lower bound of the input range.
      upper_bound (float): The upper bound of the input range.
      title (str): The title for the plot.
  """
  # Generate denser data points for smoother function visualization
  dense_X = np.linspace(lower_bound, upper_bound, 1000)
  dense_y = function(dense_X)

  plt.figure(figsize=(8, 6))
  plt.plot(X, y, 'o', label='Generated Data')
  plt.plot(dense_X, dense_y, label='Function')
  plt.xlabel('X')
  plt.ylabel('y')
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()

# Example usage - Quadratic Function
def square_function(x):
  return x**2

n_samples = 100
lower_bound = -5
upper_bound = 5

X, y = generate_data(square_function, n_samples, lower_bound, upper_bound)
visualize_data(X, y, square_function, lower_bound, upper_bound, title='Quadratic Function')

# Example usage - Rosenbrock Function



