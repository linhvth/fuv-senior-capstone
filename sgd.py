"""
Main code for Vanilla SGD

Linh Vu (2024)
"""

### IMPORT LIBRARIES
import numpy as np
class SGD:
    """
    Base class for Stochastic Gradient Descent (SGD) optimizers.

    This class provides a framework for performing SGD updates on model parameters
    based on a learning rate and gradient information. Subclasses can implement
    pecific update rules like standard SGD and noisy SGD.
    """
    def __init__(self):
        self.theta = None
        self.init_theta = None
        self.n_params = None
        self.df = None
        self.f = None
        self.X = None
        self.y = None
        self.step_size = None
        self.n_iters = None
        self.n_streams = None
        self.update_method = None
        self.strategy = None
        pass

    def get_theta(self):
        """
        Returns the optimal parameters (weights and bias) found by the optimizer.
        Returns:
            A NumPy array containing the optimal parameters.
        """
        return self.theta

    def fit(self, f, df, init_theta=None, M=None, y=None, n_iters=1000, n_streams=100, 
            step_size=0.01, update_method='avgLastPoint'):
        """
        Fits the model using SGD optimization.

        Args:
            f: The objective function to be minimized.
            df: The gradient function of the objective function.
            X: A 2D NumPy array containing the training data features.
            y: A 1D NumPy array containing the target labels.
            init_theta: A NumPy array containing initial parameters (weights) 
                        (default: None - random initialization).
            n_iter: An integer specifying the number of training iterations 
                    (default: 1000).
            n_streams: An integer specifying the number of data streams for potential 
                        parallelization (default: 1).
            update_method: A string specifying the update method (default: 'standard_sgd').
            
        Returns:
            None
        """
        # Record step_size value
        self.step_size = step_size
        
        # Check for valid data shapes
        if (M is not None) and (y is not None):
            if len(M.shape) != 2 or len(y.shape) != 1 or M.shape[0] != y.shape[0]:
                raise ValueError("Incompatible data shapes for M and y")
            self.M = M
            self.y = y
            self.n_params = M.shape[1]
        
        # Initialize theta (params) random initialization)
        if init_theta is None:
            self.init_theta = np.random.rand(self.n_params)
        else:
            self.init_theta = init_theta

        # Import information of main function and its derivative
        self.f = f
        self.df = df

        # Import other args of optimization process
        self.n_iters = n_iters
        self.n_streams = n_streams

        # Optimization process - pick either avgLastPoint or avgEachIter
        # Choose optimization strategy based on update_method
        if update_method == 'avgEachIter':
            self.optimize = self._avg_each_iter
        elif update_method == 'avgLastPoint':
            self.optimize = self._avg_last_point
        else:
            raise ValueError("Invalid update_method. Choose 'avgEachIter' or 'avgLastPoint'")
        
        # Perform SGD optimization
        self.optimize()

    def _sgd_update(self, theta, update_vector):
        """
        Updates parameters based on the learning rate and gradients.

        Args:
            params: A NumPy array containing model parameters (weights and bias).
            grad: A NumPy array containing gradients for each parameter.

        Returns:
            A NumPy array containing the updated parameters.
        """
        return theta - self.step_size*update_vector
    
    def _uniform_select_sample(self):
        # Randomly select a data point index
        index = np.random.randint(self.M.shape[0])
        # Return random pair (point)
        return self.M[index, :], self.y[index]

    def _avg_each_iter(self):
        """
        Performs SGD with update averaged across gradients from each stream 
        at every iteration.
        """
        final_theta = self.init_theta.copy()
        for _ in range(self.n_iters):
            update_vector = np.zeros_like(final_theta)
            for _ in range(self.n_streams):
                # Get random data point and its label
                m_i, y_i = self._uniform_select_sample()
                update_vector += self.df(m_i, y_i, final_theta)
            
            # Average gradients across streams
            update_vector /= self.n_streams  
            final_theta = self._update(final_theta, update_vector)

        self.theta = final_theta
                       
    def _avg_last_point(self):
        """
        Performs SGD with update averaged across final theta values from each stream.
        """
        record_last_thetas = np.zeros((self.n_streams, self.n_params))

        for i in range(self.n_streams):
            theta = self.init_theta.copy()
            for _ in range(self.n_iters):
                # Get a random vector
                m_i, y_i = self._uniform_select_sample()
                this_grad = self.df(m_i, y_i, theta)

                # update theta
                theta = self._sgd_update(theta, this_grad)
            record_last_thetas[i] = theta
        
        # Average the last thetas of all streams to obtain the final 'optimal' theta
        self.theta = np.mean(record_last_thetas, axis=0)
