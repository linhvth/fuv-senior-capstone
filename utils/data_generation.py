"""
Data Generation.

Linh Vu (2024)
"""

### IMPORT LIBRARIES
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

def check_and_convert_input(input):
  """
  Checks if the input is a tuple, array, list, or integer, 
  and converts list and tuple to NumPy array, 
      passes NumPy array, and raises an error for other types.

  Args:
    input: The value to check and convert.

  Returns:
    A NumPy array representation of the input value.

  Raises:
    ValueError: If the input is not a list, tuple, array, or int.
  """

  if isinstance(input, np.ndarray):
    return input              # Pass NumPy arrays directly
  elif isinstance(input, (list, tuple)):
    return np.asarray(input)  # Convert lists and tuples to NumPy arrays
  elif isinstance(input, int):
    return np.array([input])  # Convert int to a NumPy array with 1 element
  else:
    raise ValueError(f"Invalid input type: {type(input)}. Expected list, tuple, array, or int.")

def generate_data(f, n_samples, lower_bounds, upper_bounds):
  """
  Generates data from a given multivariable function.

  Args:
      func: The multivariable function to generate data from.
      lower_bounds: A list of lower bounds for each variable.
      upper_bounds: A list of upper bounds for each variable.
      num_samples: The number of data samples to generate.

  Returns:
      A numpy array of shape (num_samples, d) where d is 
      the number of variables.
  """
  lower_bounds = check_and_convert_input(lower_bounds)
  upper_bounds = check_and_convert_input(upper_bounds)

  # Generate random samples within the specified bounds
  X = np.random.rand(n_samples, len(lower_bounds))
  for i in range(len(lower_bounds)):
    X[:, i] = X[:, i] * (upper_bounds[i] - lower_bounds[i]) + lower_bounds[i]

  noise = np.random.normal(loc=0, scale=random.randint(5), size=(n_samples, 1))
  
  # Evaluate the function on the generated samples
  y = f(X) + noise

  return X, y.reshape(n_samples, -1)


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

if __name__ == "__main__":
  # Example usage - Quadratic Function
  def square_function(x):
    return x**2

  n_samples = 500
  lower_bound = -5
  upper_bound = 5

  X, y = generate_data(square_function, n_samples, 
                      lower_bound, upper_bound)

  visualize_data(X, y, square_function, lower_bound, 
                upper_bound, title='Quadratic Function')
