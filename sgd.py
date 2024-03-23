"""
Main code for Vanilla SGD

Linh Vu (2024)
"""

### IMPORT LIBRARIES
import numpy as np

class SGD:
    def __init__(self):
        """
        Stochastic Gradient Descent optimizer.
        """
        self.f = None
        self.df = None
        self.n_dims = None
        self.step_size = None
        self.max_iter=None
        self.epsilon = None
        self.tolerance = None
        self.max_iter = None
        self.init_x = None
        self.x = None
    
    def _update(self, curr_x, stochastic_vec):
        """
        Performs a single update step of the SGD algorithm.
        """
        df_prev_x = self.df(curr_x)
        self.x -= self.stepsize * (df_prev_x + stochastic_vec)

    def optimize(self, f, df, n_dims, init_point=None, step_size=1e-2, 
                 max_iter=1000, max_stream=100, epsilon=1e-2, tolerance=1e-4):
        """
        Optimizes the function using SGD for a maximum number of iterations.

        Args:
            f: The function to optimize (objective function).
            df: The gradient function of f.
            n_dims: The dimensionality of the input.
            init_point: The initial guess for the optimal point (default: None, random initialization).
            step_size: The learning rate for gradient updates (default: 1e-2).
            max_iter: The maximum number of iterations to perform (default: 1000).
            epsilon: neighborhood
            tolerance: The convergence tolerance (default: 1e-4).

        Returns:
        The optimal point found by SGD.
        """
        ### Setup class's params
        self.f = f
        self.df = df
        self.n_dims = n_dims
        self.init_x = init_point if init_point is not None else np.random.rand(self.n_dims)
        self.step_size = step_size
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.n_streams = max_stream

        
        ### Performs SGD update in each stream, then take the average of all final x values 
        ### obtained from whose streams.
        record_last_thetas = np.zeros((self.n_streams, self.n_params))
        for i in range(self.n_streams):
            x = self.init_x.copy()
            for _ in range(self.max_iter):
                # Get a random vector
                noise = self._gaussian_noise()
                # Update theta
                x = self._update(x, noise)
                if np.linalg.norm(self.df(x)) < self.tolerance:
                    break
            # Store values of lasts point of each stream
            record_last_thetas[i] = x
        
        # Average the last thetas of all streams to obtain the final 'optimal' theta
        self.x = np.mean(record_last_thetas, axis=0)
        
    def get_solution(self):
        return self.x
    
    def _gaussian_noise(self):
        return np.random.normal(loc=0, scale=self.epsilon, size=self.n_dims)
                       
    def _avg_last_point(self):
        """
        Performs SGD with update averaged across final theta values from each stream.
        """
        record_last_thetas = np.zeros((self.n_streams, self.n_params))
        for i in range(self.n_streams):
            x = self.init_x.copy()
            for _ in range(self.max_iter):
                # Get a random vector
                noise = self._gaussian_noise()
                # Update theta
                x = self._update(x, noise)
            # Store values of lasts point of each stream
            record_last_thetas[i] = x
        
        # Average the last thetas of all streams to obtain the final 'optimal' theta
        self.x = np.mean(record_last_thetas, axis=0)
