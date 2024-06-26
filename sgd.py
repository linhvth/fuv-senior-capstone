"""
Main code for Vanilla SGD

Linh Vu (2024)
"""

### IMPORT LIBRARIES
import copy
from math import sqrt
import numpy as np

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This class implements the SGD algorithm for optimization. It allows you to
    control hyperparameters and run multiple independent SGD streams for
    experimentation.
    """

    def __init__(self, stepsize_schedule, step_size, max_iter=1000, 
                 n_streams=30, tolerance=1e-6):
        """
        Initializes the SGD optimizer with default hyperparameters.

        Args:
            step_size (float, optional): The learning rate for gradient updates.
                Defaults to 1e-2.
            max_iter (int, optional): The maximum number of iterations to perform
                in each SGD stream. Defaults to 1000.
            epsilon (float, optional): Neighborhood parameter (used internally).
                Defaults to 1e-2.
            tolerance (float, optional): The convergence tolerance. Defaults to
        """
        # Init characteristics of the function f
        self.f = None               # function f
        self.df = None              # derivative of f
        self.n_dims = None          # no. dimensions
        self.Lipshitz_g = None      # Lipschitz constant of the NOISE VECTOR g(x, epsilon)

        if stepsize_schedule not in ['fixed', 'decreasing']:
            raise ValueError("stepsize_schedule must be either 'fixed' or 'decreasing'.")
        self.stepsize_schedule = stepsize_schedule
        self.step_size = step_size  

        self.max_iter = max_iter
        self.tolerance = tolerance
        self.n_streams = n_streams
        self.early_stopping = False

        # Noise Distribution - assumed to be Gaussian
        self.mu = 0             # init
        self.noise_std = 1      # init value
        
        # Init to record result of iterations
        self.init_x = None
        self.x = None
        self.record_last_x = None
        self.convergence_history_all_streams = dict()
        self.convergence_history_average = None

    def update_hyperparams(self, **kwargs):
        """
        Updates hyperparameters of the optimizer.

        This method allows you to modify hyperparameters of the SGD optimizer after
        it has been created. You can provide any combination of the following arguments
        to update their corresponding values.

        Args:
            step_size (float, optional): The learning rate for gradient updates.
            max_iter (int, optional): The maximum number of iterations to perform
                in each SGD stream.
            n_streams (int, optional): The number of independent SGD streams to run.
            epsilon (float, optional): Neighborhood parameter (used internally).
            tolerance (float, optional): The convergence tolerance.
        """

        # Update only provided arguments using unpacking
        for key, value in kwargs.items():
            if key in ("step_size", "max_iter", "n_streams", "epsilon", "tolerance", "early_stopping"):
                setattr(self, key, value)
            
    
    def _update(self, curr_x, stochastic_vec, step_size):
        """
        Performs a single update step of the SGD algorithm.
        """
        df_prev_x = self.df(curr_x)
        updated_x = curr_x - step_size * (df_prev_x + stochastic_vec)
        return updated_x
    
    def _stepsize_schedule_update(self, iter):
        if self.stepsize_schedule == 'decreasing':
            return copy.deepcopy(self.step_size) / sqrt(iter)
        elif self.stepsize_schedule == 'fixed':
            return self.step_size
        
    def optimize(self, f, df, n_dims, init_point=None):
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

        ### Setup data record
        self.record_last_x = np.zeros((self.n_streams, self.n_dims))
        sum_all_streams = None

        ### Performs SGD update in each stream, then take the average of all final x values 
        ### obtained from whose streams.
        for i in range(self.n_streams):
            this_convergence_history = list()
            x = copy.deepcopy(self.init_x) # reset x to init_point for each stream

            for j in range(self.max_iter):
                # Get a random vector
                noise = self._gaussian_noise()
                # Update theta
                step_size = self._stepsize_schedule_update(j+1)
                x = self._update(x, noise, step_size)

                if (self.early_stopping) & (np.linalg.norm(self.df(x)) < self.tolerance):
                    break

                # Record convergence history after each 10 iterations
                this_convergence_history.append(x)
                # if j % 5 == 0:
                #     this_convergence_history.append(x)

            # Store values of lasts point of each stream
            self.record_last_x[i] = x

            # Add convergece history of this stream to record
            if i != 0:
                sum_all_streams = [a + b for a, b in zip(sum_all_streams, this_convergence_history)]

            else:
                sum_all_streams = this_convergence_history

            self.convergence_history_all_streams[f'k={i+1}'] = this_convergence_history
        
        # Average the last thetas of all streams to obtain the final 'optimal' theta
        self.x = np.mean(self.record_last_x, axis=0)

        # Compute average data for convergence history
        self.convergence_history_average = [arr / self.n_streams for arr in sum_all_streams]

    def get_convergence_history_all_streams(self):
        return self.convergence_history_all_streams
    
    def get_convergence_history_average(self):
        return self.convergence_history_average
        
    def get_init_point(self):
        return self.init_x
    
    def get_last_points(self):
        return self.record_last_x
    
    def get_solution(self):
        return self.x
    
    def noise_gaussian_setup(self, mu=0, sigma=1) -> None:
        self.mu = mu
        self.noise_std = sigma

    def _gaussian_noise(self):
        return np.random.normal(loc=self.mu, 
                                scale=self.noise_std, 
                                size=self.n_dims)
    
                       
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
