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
from sgd import *

class LogisticRegression:
    """
    This implementation assume that the value of y is either 0 or 1
    """
    def __init__(self):
        self.theta = None      # placeholder for params
        self.optimizer = None  # optimizer
    
    def fit(self, M, y, optimizer, n_iters=1000, n_streams=100, step_size=0.01, 
            update_method='avgLastPoint'):
        init_theta = np.random.randn(M.shape[1])

        self.optimizer = optimizer
        self.optimizer.fit(f=self._loss_function_single, df=self._loss_gradient_single, 
                           init_theta=init_theta, M=M, y=y, n_iters=n_iters, n_streams=n_streams, 
                           step_size=step_size, update_method=update_method)
        
        # Get optimal theta from optimizer
        self.theta = optimizer.get_theta()
    
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
    
    def _loss_function_single(self, m, y, theta):
        z = theta.T@m
        return - y*z + np.log(1 + np.exp(z))
    
    def _loss_gradient_single(self, m, y, theta):
        z = theta.T@m
        return (-y + 1/(1+np.exp(-z)))*m