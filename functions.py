"""
This module provides functions for generating data for Stochastic Gradient 
Descent (SGD) testing.

These functions offer tools to create datasets with specific characteristics 
suitable for evaluating SGD algorithms.
"""

### Test 1: Simple square functions
f = lambda x: x**2
df = lambda x: 2*x

### Test 2: Rosenbrock 
rosenbrock = lambda X, b: (X[0]-1)**2 + b*(X[1]-X[0]**2)**2



