"""
Noisy gradient
"""

import numpy as np

class GaussianNoisyGradient():
    def __init__(self, step_size=0.01, noise_std=1.0):
        super().__init__()
        self.step_size = step_size
        self.noise_std = noise_std

    def noisy_gradient_update(self, theta, full_grad, dist="gaussian"): # this is not correct, will edit later
        """
        SGD update with added noise.
        """
        if dist == 'gaussian':
            noise = np.random.normal(loc=0, scale=self.noise_std, size=theta.shape) 
            theta -= self.step_size * (full_grad + noise)
        return theta