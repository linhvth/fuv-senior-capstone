"""
Description: 
    This Python file implements a collection of simple machine learning 
    models for experimentation purposes. These models can be used as building 
    blocks for more complex architectures or for exploring various algorithms 
    on a particular dataset.

Models: 
    - Perceptron            for non-convex loss function
    - Logistic Regression   for convex loss function

Author: Linh Vu (2024)
"""

import numpy as np

class LogisticRegression:
    """
    This implementation assume that the value of y is either 0 or 1
    """
    def __init__(self, optimizer) -> None:
        self.theta = None           # placeholder for params
        self.optimizer = optimizer  # optimizer
    
    def fit(self, M, y, optimizer):

        return 
    
    def predict(self, M, threshold=0.5):
        """
        Predict the label of y (0 or 1) given information X
        Input:
            M (np.array): matrix of information
        Return:
            a list of predicted labels (np.array) where each 
                element is either 0 or 1
        """
        z = M@self.theta
        y_pred = self._sigmoid_function(z)
        return np.where(y_pred >= threshold, 1, 0)

    def _sigmoid_function(self, value):
        return 1/(1 + np.exp(-value))
    
    def _loss_function_single(self, m, y):
        z = self.theta.T@m
        return - y*z + np.log(1 + np.exp(z))
    
    def _loss_gradient_single(self, m, y):
        z = self.theta.T@m
        return (-y + 1/(1+np.exp(-z)))*m