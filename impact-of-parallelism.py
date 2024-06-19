import inspect
import os
import copy
import time
from datetime import datetime

import numpy as np
import pandas as pd

from SGD import *
from utils.data_generation import *
from CONSTANTS import *
from utils.misc import *
from utils.functions import *
import pickle


# def testing_opt(f, df, n_dims: int, true_global_min: int, init_point=None, 
#                 step_size=1e-2, max_iter=1000, n_streams=300, epsilon=1e-3, 
#                 tolerance=1e-20, save_result=True, name_result=None, 
#                 print_terminal=True):
#     optimizer = SGD()
#     optimizer.optimize(f, df, n_dims=n_dims, init_point=init_point, step_size=step_size, 
#                         max_iter=max_iter, n_streams=n_streams, epsilon=epsilon, 
#                         tolerance=tolerance)
    
#     # Create result map
#     result = dict()

#     result['f'] = inspect.getsource(f).split("=", 1)[1].split(":", 1)[1][:-1]
#     result['true solution'] = true_global_min
#     result['init_point'] = optimizer.get_init_point()
#     result['solution found by SGD'] = optimizer.get_solution()

#     if save_result:
#         # Create path to record result
#         get_date = datetime.date.today().strftime('%Y%m%d')
#         check_path_file(RESULT_PATH)
#         filename = name_result if name_result is not None else '_untitled'
#         save_path = os.path.join(RESULT_PATH, get_date + filename + '.txt')

#         # Write to the result file
#         write_as_txt(result, save_path)

#     if print_terminal:
#         print(result)
#         print(f"f: {f(optimizer.get_solution())}")
#         print(f"df: {df(optimizer.get_solution())}")


# Use this function for each number of streams
def _helper_each_k(f, df, n_dims, optimizer, init_point, n_streams, n_sample):
    """
    For each running time
    """
    # Update n_streams for the specific case of experiments
    optimizer.update_hyperparams(n_streams=n_streams)
    collected_sample = list()

    for _ in range(n_sample): # repeat to collect realizations
        init_copy = copy.deepcopy(init_point) # reset to init_point for each unit of sample
        optimizer.optimize(f, df, n_dims=n_dims, init_point=init_copy) 
        last_xs = optimizer.get_last_points()
        final_x = np.round(optimizer.get_solution(), decimals=4)
        collected_sample.append(final_x)
    
    return last_xs, collected_sample


def impact_of_parallelism(f, df, n_dims, optimizer, init_point, Ks, n_sample=100, 
                          save_result=False, name_result=None):
    data = dict()

    for k in Ks: # for each no. streams
        print(f'no. streams = {k}')
        start_time = time.time()
        _, this_data =_helper_each_k(f, df, n_dims, optimizer, 
                                     init_point=copy.deepcopy(init_point), 
                                     n_streams=k, n_sample=n_sample)
        data[f'k={k}'] = this_data
        print("--- %s seconds ---" % (time.time() - start_time))
    
    data = pd.DataFrame.from_dict(data)

    if save_result:
        # Create path to record result
        get_date = datetime.date.today().strftime('%Y%m%d')
        check_path_file(RESULT_PATH)
        filename = name_result if name_result is not None else '_untitled'
        save_path = os.path.join(RESULT_PATH, filename + '.pkl')

        # Write to the result file
        write_to_pkl(data, save_path)

    return


if __name__ == '__main__':
    Ks = [1, 10, 50, 100]
    n_ind_runs = 50

    # Setup an example to conduct experiments
    n_dims = 5 # this is just an example, can be replaced by any natural number
    quad = simple_quadratic_func(n_dims=n_dims)
    init_point = np.array([2, -3, 5, -6, 1])
    D = np.linalg.norm(init_point - quad.global_min)
    B = 1

    # Setup init optimizer for consistency
    optimizer = SGD()   # init optimizer
    optimizer.noise_gaussian_setup(mu=0, sigma_square=B**2)

    # Setup some params for this optimizer
    optimizer.update_hyperparams(step_size = D/B, max_iter = 1000,)

    
    print(D/B)
    # start_time = time.time()
    data = impact_of_parallelism(f=quad.function, df=quad.derivative, n_dims=n_dims, 
                                 optimizer=optimizer, init_point=init_point.copy(), 
                                 Ks=Ks, n_sample=n_ind_runs, save_result=True,
                                 name_result='quad_parallelism')
    
    df = pd.read_pickle('results/quad_parallelism.pkl')
    write_as_csv(df, 'results/quad_parallelism.csv')
