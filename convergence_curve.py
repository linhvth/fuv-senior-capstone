import os, re, copy, time, inspect, pickle
from datetime import date

import math
import numpy as np
import pandas as pd

from SGD import SGD
import CONSTANTS
from utils.data_generation import *
from utils import functions, misc

###
exp_result_folder_path = os.path.join(CONSTANTS.RESULT_PATH, 'experiment_1_quad_m=2')
misc.check_path_file(exp_result_folder_path)
plot_folder_path = os.path.join(exp_result_folder_path, 'plotting')
misc.check_path_file(plot_folder_path)

  
 
def case_3_convergence_curve(k, alpha, B, D, L_g, m):
    return (1 + (alpha**2)*(L_g**2) - 2*m*alpha)**k * D**2 \
        + (alpha* (B**2))/(2*m - alpha* (L_g)**2)

def experiment_quad(n_dims, init_point, noise_sigma, step_size='rand', 
                    n_streams=1, n_iters=100, save_result=False):
    "This function is dedicated for quad function."
    # get function information with the input of number of dimensions
    quad = functions.simple_quadratic_func(n_dims=n_dims) 

    if init_point.shape[0] != n_dims:
        raise ValueError(f"Starting point has incorrect dimensions. \
                         Expected {n_dims} dimensions, \
                         but got {init_point.shape[0]}.")

    SIGMA = noise_sigma # std of Gaussian distribution for additive random noise \xi
    D = np.linalg.norm(init_point - quad.global_min)
    B = math.sqrt(quad.n_dims) * SIGMA
    M = quad.strong_convexity_modulus
    LG = quad.lipschitz_constant_df

    upper_bound_alpha = 2*M/(LG**2)

    if step_size == 'rand':
        step_size = random.uniform(low=0, high=upper_bound_alpha)

    optimizer = SGD(stepsize_schedule='fixed', step_size=step_size, 
                    max_iter=n_iters, n_streams=n_streams)
    optimizer.noise_gaussian_setup(sigma=SIGMA)
    optimizer.optimize(f=quad.function, df=quad.derivative, n_dims=quad.n_dims, 
                       init_point=init_point)
    
    convergence_history_all_streams = optimizer.get_convergence_history_all_streams()
    convergence_history_avg = optimizer.get_convergence_history_average()

    # save data for future reference
    if save_result:
        # Create path to record result
        get_date = date.today().strftime('%Y%m%d')
        misc.check_path_file(CONSTANTS.RESULT_PATH)
        filename = get_date + '_' + 'convergence_history_quad'
        save_path = os.path.join(CONSTANTS.RESULT_PATH, filename + '.csv')

        # Create DataFrame from dictionary
        df = pd.DataFrame.from_dict(convergence_history_all_streams)
        # Save DataFrame to CSV file
        df.to_csv(save_path, index=False) 


    # norm of distance between each iterates to true solution
    convergence_error_avg = [(np.linalg.norm(quad.global_min - iterates))**2 \
                             for iterates in convergence_history_avg]
    error_0 = np.linalg.norm(quad.global_min - init_point)**2
    convergence_error_avg.insert(0, error_0)

    # theoretical curve
    # iters = np.array([x for x in range(1, n_iters+1) if x % 5 == 1])
    iters = np.array([x for x in range(0, n_iters+1)])
    convergence_theory = case_3_convergence_curve(k=iters, alpha=step_size, 
                                                  B=B, D=D, L_g=LG, m=M)

    return convergence_error_avg, convergence_theory, iters

def experiment_1(init_point, noise_sigma, step_size, n_iters, n_runs=30, save_result=False):
    ### Experiment 1: n_streams = 1
    n_dims = init_point.shape[0]
    step_size = step_size
    empirical_data = dict()

    for i in range(n_runs):
        avg_path , theoretical_curve, iters = \
            experiment_quad(n_dims=n_dims, init_point=init_point, noise_sigma=noise_sigma, 
                            step_size=step_size, n_streams=1, n_iters=n_iters)

        empirical_data[f'ind_run_{i+1}'] = avg_path
    
    data = pd.DataFrame.from_dict(empirical_data)
    empirical_curve = data.mean(axis=1)
    data['expected_error'] = empirical_curve
    data['theoretical_bound'] = theoretical_curve

    if save_result:
        # Create path to record result
        get_date = date.today().strftime('%Y%m%d')
        filename = f"{get_date}_stepsize={step_size}_gaussianSigma={noise_sigma}"
        save_path = os.path.join(exp_result_folder_path, filename + '.csv')
        data.to_csv(save_path, index=False)

    return data, iters

def plot_convergence_curve(iters, single_run, empirical_data, theoretical_data, 
                           filename, figure_path):
    superscript_2 = chr(0x00B2)  # Unicode character for superscript 2
    norm = r"||"  # Raw string for norm symbol (double vertical bars)

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Adjust figsize as needed

    # Create plots on each subplot
    ax1.plot(iters, single_run, color='green', label='Emprical (Single run)')
    ax1.plot(iters, empirical_data, color='red', label='Emprical (Averaged)')
    ax1.plot(iters, theoretical_data, color='blue', linestyle='--', label='Theoretical')
    ax1.set_xlabel('Iteration k')
    ax1.set_ylabel("Distance to Minimum (Expected Squared Norm)")
    ax1.set_title(f'Convergence History: {filename}')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(iters[0:100], single_run[0:100], color='green', label='Emprical (Single run)')
    ax2.plot(iters[0:100], empirical_data[0:100], color='red', label='Emprical (Averaged)')
    ax2.plot(iters[0:100], theoretical_data[0:100], color='blue', linestyle='--', label='Theoretical')
    ax2.set_xlabel('Iteration k')
    ax2.set_ylabel("Distance to Minimum (Expected Squared Norm)")
    ax2.set_title(f'Convergence History (Zoom in): {filename}')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout (optional)
    plt.tight_layout()

    # Save the plot
    plt.savefig(figure_path)
    plt.clf() # clear plot env to avoid stacking

def experiment1_plotting(filepath):
    data = pd.read_csv(filepath)

    # randomly pick an independent single run
    single_empirical_data = data.iloc[:, 35]

    # get empirical and theoretical data
    empirical_data = data['expected_error']
    theoretical_data = data['theoretical_bound']
    iters =  np.array([x for x in range(1, data.shape[0]+1)])


    # Regular expression pattern to capture stepsize and noise info
    pattern = r"(?:^|_)(stepsize=[\d.]+_gaussianSigma=[\d.]+)(?:$|\.)" 
    # Get filename for this plot
    unique_name = re.search(pattern, filepath).group(1)
    figure_name = f"{unique_name}.png"
    figure_path = os.path.join(plot_folder_path, figure_name)

    plot_convergence_curve(iters, single_empirical_data, 
                           empirical_data, theoretical_data,
                           unique_name, figure_path)

if __name__ == '__main__':
    ### Experiment 1: n_streams = 1

    # WARNING: ALREADY EXECUTED THESE BELOW LINES AND RECORDED DATA
    init_point = np.array([-4, 2, 4, 8])
    noise_sigmas = [1, 5, 10]
    step_sizes = [0.01, 0.1, 0.5, 0.95]
    n_iters = 1000
    n_runs = 100

    combinations = [(step_size, noise) for step_size in step_sizes for noise in noise_sigmas]
    for step_size, noise_sigma in combinations:
        data, iters = experiment_1(init_point, noise_sigma, step_size, 
                                   n_iters, n_runs, save_result=True)

    for filename in os.listdir(exp_result_folder_path):
        if filename != 'plotting':
            filepath = os.path.join(exp_result_folder_path, filename)
            experiment1_plotting(filepath)
    
    # PLOTTING # zoom in [0:50]
    # single_empirical_data = data.iloc[:, 3]
    # empirical_data = data['expected_error']
    # theoretical_data = data['theoretical_bound']
    # iters =  np.array([x for x in range(1, data.shape[0]+1)])
    # plt.plot(iters[0:100], single_empirical_data[0:100], color='green', label="single record")
    # plot_convergence_curve(iters[0:100], empirical_data[0:100], theoretical_data[0:100])

    # plt.plot(iters, single_empirical_data, color='green', label="single record")
    # plot_convergence_curve(iters, empirical_data, theoretical_data)


    ### Experiment 2
    