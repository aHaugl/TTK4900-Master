"""
@author: Andreas Haugland

Based on project thesis from spring 2021 by Andreas Haugland, in
turn based on ESKF project in TTK4550 fall 2020

"""

# %% 
from plotter import plot_NEES, plot_NIS, plot_angle, plot_error_pos_sigma, plot_error_vel_sigma, plot_path, state_error_plots
from typing import Tuple, Sequence, Any
from matplotlib import pyplot as plt


# from plotfunctionstesting import plot_acc, plot_acc2, plot_gyro, plot_angle

import numpy as np
import scipy.linalg as la
import scipy
import scipy.io
import scipy.stats
from dataclasses import dataclass, field
import timeit
import time

import tqdm
import os
import sys

from tqdm import trange
from tqdm import tqdm_notebook

from quaternion import * 
from cat_slice import CatSlice
# from eskf import ESKF
# from eskf import ESKF

from utils import UDU_factorization

from eskf_batch import ESKF_batch
from eskf_sequential import ESKF_sequential
from eskf_UDU import ESKF_udu
from eskf_runner import run_batch_eskf, run_sequential_eskf, run_UDU_eskf
from plotter import * #plot_error_v_sigma, plot_pos, plot_vel, plot_angle, plot_estimate, plot_3Dpath, plot_path, state_error_plots, plot_NEES, plot_NIS
# from timer import * 


# %% plot config check and style setup

plt.close('all')

from plot_setup import setup_plot
setup_plot()
np.seterr(all='raise')
scipy.special.seterr(all='raise')


# %% Q and R matrixes are defined as
"""
Q_err: acc_std, gyro_std, acc_bias_Std, gyro_bias std
R: p_std -> R_GNSS = p_std**2*eye(3)
"""

# %% Load data and plot 
"""
Loads true state, time steps and other simulation variables. Run gen_mat to generate
new data
"""

folder = os.path.dirname(__file__)
# filename_to_load = f"{folder}/../data/simulation_params.mat"
# filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver.mat"
# filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver_long.mat"
# filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver_long_ver2.mat"
filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver_long_ver3.mat"
cache_folder = os.path.join(folder,'..', 'cache')
loaded_data = scipy.io.loadmat(filename_to_load)

timeIMU = loaded_data["timeIMU"].ravel()
#acc_t = loaded_data["acc_t"].T
if 'x_true' in loaded_data:
    x_true = loaded_data["x_true"].T
else:
    x_true = None
# z_GNSS = loaded_data["z_GNSS"].T
    
beacon_location = loaded_data["beacon_location"]

tGNSS = loaded_data["timeGNSS"].T
z_GNSS = loaded_data["z_GNSS"].T

z_acc_vector = loaded_data["z_acc"].T
acc_t = loaded_data["acc_t"].T
omega_t = loaded_data["omega_t"].T
z_gyro_vector = loaded_data["z_gyro"].T

dt = np.mean(np.diff(timeIMU))



# %% indices and initialization
POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)

ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)

# %% 

# Initializing prediction
x_pred_init = np.zeros(16)

# 8 figure init
x_pred_init[POS_IDX] = np.array([10,0,1])
x_pred_init[VEL_IDX] = np.array([0, 2, 0])
x_pred_init[6:10] = np.array([1,0,0,0]) #(0,0,0) iny euler
# x_pred_init[10:13] = np.array([-8.75e-2,-8.74e-2,6.9e-2])
# x_pred_init[13:16] = np.array([-9.54e-4,-7.32e-04,3.37e-04])

# x_pred_init[POS_IDX] = np.array([0, 0, -5])  # starting 5 metres above ground
# x_pred_init[VEL_IDX] = np.array([20, 0, 0])  # starting at 20 m/s due north
# no initial rotation: nose to North, right to East, and belly down
# x_pred_init[6] = 1


# %% From another simulation

# %% Q matrix params
# acc_std = 7.59562072e-02
# rate_std =  5.12679127e-02

# cont_acc_bias_driving_noise_std =  1.16562328e-04
# cont_rate_bias_driving_noise_std =  1.94631260e-05

# %% Current best
# TUNABLE0.01035
# cont_gyro_noise_std =  0.01035
cont_gyro_noise_std =   4.36e-5  # (rad/s)/sqrt(Hz)
# TUNABLE 0.015191 
# cont_acc_noise_std = np.array([0.01, 0.01, 0.042191])
# cont_acc_noise_std = 0.01549 #1.167e-2  # (m/s**2)/sqrt(Hz)
cont_acc_noise_std =  1.167e-2  # (m/s**2)/sqrt(Hz)

# cont_gyro_noise_std = 0.001      #4.36e-5  # (rad/s)/sqrt(Hz)
# cont_acc_noise_std = 0.001  #1.167e-3  # (m/s**2)/sqrt(Hz)

# Discrete sample noise at simulation rate used
# This is the formula from example 4.9. Should this be the driving noise instead?
rate_std = np.round(0.5 * cont_gyro_noise_std * np.sqrt(1 / dt),7) #Just to avoid float shit
acc_std = np.round(0.5 * cont_acc_noise_std * np.sqrt(1 / dt),7)

# Bias values
# TUNABLE 0.0005839  0.00005839 
# 5e-5 From another simulation
rate_bias_driving_noise_std =  5e-5 

cont_rate_bias_driving_noise_std = (
    np.round((1 / 3) * rate_bias_driving_noise_std / np.sqrt(1 / dt), 7)
)
#4e-3 From another simulation
#0.0001947, 0.0020947 0.0050947
acc_bias_driving_noise_std = 4e-3
cont_acc_bias_driving_noise_std = 6 * acc_bias_driving_noise_std / np.sqrt(1 / dt)

# %% Used to approx acc bias and gyro bias >0
p_acc =   1.000e-6
p_gyro = 1.00000000e-6

# %% Position and velocity measurement, Used in R
# p_std = np.array([1.58170828, 1.58170828, 0.97243219])   # Measurement noise

#The measurements from the GNSS are perfect at the moment
# p_std = np.array([0, 0, 0])
p_std = np.array([.01, .01, .03])
# p_std = np.array([.1, .1, .3])
# p_std = np.array([.3, .3, .5])

# %% Cov matrix initalization

P_pred_init = np.zeros((15,15))
P_pred_init[POS_IDX ** 2] = np.eye(3) * 3**2  
P_pred_init[VEL_IDX ** 2] = np.eye(3) * 2**2  #0.2
P_pred_init[ERR_ATT_IDX ** 2] = np.eye(3) * 0.5**2  # TODO
P_pred_init[ERR_ACC_BIAS_IDX ** 2] = np.eye(3) * 0.01**2 
P_pred_init[ERR_GYRO_BIAS_IDX ** 2] = np.eye(3) * 0.001**2 # 0.001**2 
 
eskf_parameters = [acc_std,
                    rate_std,
                    cont_acc_bias_driving_noise_std,
                    cont_rate_bias_driving_noise_std,
                    p_acc,
                    p_gyro]   

"""
Running the simulation
"""
beacon_location: np.ndarray = loaded_data["beacon_location"]

use_batch_pseudoranges: bool = True
# use_batch_pseudoranges: bool = False
use_sequential_pseudoranges: bool = True
# use_sequential_pseudoranges: bool = False
use_UDU: bool = True
# use_UDU: bool = False

num_beacons = len(beacon_location)
num_sims = 2

# %% Run estimation for
#Number of seconds of simulation to run. len(timeIMU) decides max
N_list: int = [int(10/dt), int(50/dt)]#, int(100/dt), int(600/dt), int(1000/dt)]
# N_list: int = [int(1000/dt)]
# N: int = int(10/dt) 
# N: int = int(50/dt)
# N: int = int(600/dt) 
# N: int = int(1000/dt)
# N: int = int(90000)
# N: int = len(timeIMU)
for i in range(len(N_list)): 
    N = N_list[i]
        
    if(N == 1000/dt):
        num_sims = 20
    print("N is " ,N)
    print("Duration is: ",N*dt)
    offset = 0
    doGNSS: bool = True
    # rtol = 1e-05
    # atol = 1e-08

    t = np.linspace(0,dt * (N-1), N)


    #Timers for batch-filter
    t_batch = np.zeros(num_sims)
    elapsed_batch = np.zeros(num_sims)

    #Timers for sequential filter
    t_sequential = np.zeros(num_sims)
    elapsed_sequential = np.zeros(num_sims)

    #Timers for UDU filter
    t_UDU = np.zeros(num_sims)
    elapsed_UDU = np.zeros(num_sims)

    #Timers for submodules
    total_elapsed_pred_timer_batch = np.zeros(num_sims)
    total_elapsed_est_timer_batch = np.zeros(num_sims)
    average_elapsed_pred_timer_batch = np.zeros(num_sims)
    average_elapsed_est_timer_batch = np.zeros(num_sims)

    total_elapsed_pred_timer_sequential = np.zeros(num_sims)
    total_elapsed_est_timer_sequential = np.zeros(num_sims)
    average_elapsed_pred_timer_sequential = np.zeros(num_sims)
    average_elapsed_est_timer_sequential = np.zeros(num_sims)

    total_elapsed_pred_timer_UDU = np.zeros(num_sims)
    total_elapsed_est_timer_UDU = np.zeros(num_sims)
    average_elapsed_pred_timer_UDU = np.zeros(num_sims)
    average_elapsed_est_timer_UDU = np.zeros(num_sims)

    print("Timerlists have been initialized")

    
    print("\nNumber of beacons used: ", num_beacons)
    print("Number of simulations ran through", num_sims)
    print("Simulation duration (seconds): ", N*dt) 

    
    #if (use_batch_pseudoranges):
    for i in range(num_sims):  
        # timeit.timeit()
        print("\nUsing batch pseudoranges without factorization. Run number: ", i+1)
        
        t_batch[i] = time.time()
        
        (x_pred,
        x_est,
        P_est,
        GNSSk,
        est_timer,
        pred_timer,
        elapsed_pred_timer_batch,
        elapsed_est_timer_batch
        ) = run_batch_eskf (
                            N, loaded_data,
                            eskf_parameters,
                            x_pred_init, P_pred_init, p_std, 
                            num_beacons,
                            offset =0.0, 
                            use_GNSSaccuracy=False, doGNSS=True,
                            debug=False
                            )
        elapsed_batch[i] = time.time() - t_batch[i]
        #Summed up time used in prediction steps
        total_elapsed_pred_timer_batch[i] = np.sum(elapsed_pred_timer_batch)
        #Summed up time used in estimation steps
        total_elapsed_est_timer_batch[i] = np.sum(elapsed_est_timer_batch)

    
    #Plot the latest run and save figures
    if (N == 10/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '10batch', 'path_batch',x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '10batch', 'path3d_batch',x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '10batch', 'error_pos_sigma_batch')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '10batch', 'error_vel_sigma_batch')
        plot_error_att_sigma(x_est, x_true, P_est, N, '10batch', 'error_att_sigma_batch')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '10batch', 'error_acc_bias_sigma_batch')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '10batch', 'error_rate_bias_sigma_batch')
    if (N == 50/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '50batch', 'path_batch',x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '50batch', 'path3d_batch',x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '50batch', 'error_pos_sigma_batch')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '50batch', 'error_vel_sigma_batch')
        plot_error_att_sigma(x_est, x_true, P_est, N, '50batch', 'error_att_sigma_batch')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '50batch', 'error_acc_bias_sigma_batch')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '50batch', 'error_rate_bias_sigma_batch')
    if (N == 100/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '100batch', 'path_batch',x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '100batch', 'path3d_batch',x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '100batch', 'error_pos_sigma_batch')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '100batch', 'error_vel_sigma_batch')
        plot_error_att_sigma(x_est, x_true, P_est, N, '100batch', 'error_att_sigma_batch')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '100batch', 'error_acc_bias_sigma_batch')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '100batch', 'error_rate_bias_sigma_batch')
        
    if (N == 600/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '600batch', 'path_batch',x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '600batch', 'path3d_batch',x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '600batch', 'error_pos_sigma_batch')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '600batch', 'error_vel_sigma_batch')
        plot_error_att_sigma(x_est, x_true, P_est, N, '600batch', 'error_att_sigma_batch')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '600batch', 'error_acc_bias_sigma_batch')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '60batch', 'error_rate_bias_sigma_batch')
        
    if (N == 1000/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '1000batch', 'path_batch',x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '1000batch', 'path3d_batch',x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '1000batch', 'error_pos_sigma_batch')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '1000batch', 'error_vel_sigma_batch')
        plot_error_att_sigma(x_est, x_true, P_est, N, '1000batch', 'error_att_sigma_batch')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '1000batch', 'error_acc_bias_sigma_batch')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '1000batch', 'error_rate_bias_sigma_batch')
    # Batch timings
          
    print("\nEllapsed time for batch: ", np.round(elapsed_batch,3))
    print("Summed runtime used in prediction module: ", np.round(total_elapsed_pred_timer_batch,3), "seconds")
    print("Portion of runtime prediction module occupies: ", np.round(total_elapsed_pred_timer_batch/elapsed_batch*100,3), "%")
    print("Summed runtime used in estimation module: ", np.round(total_elapsed_est_timer_batch,3), "seconds")
    print("Portion of runtime estimation module occupies: ", np.round(total_elapsed_est_timer_batch/elapsed_batch*100,3), "%")

    average_time_batch = np.round(np.average(elapsed_batch),3)
    print("\nAverage time for batch elapsed: ", average_time_batch, "seconds")
    average_elapsed_pred_timer_batch = np.round(np.average(total_elapsed_pred_timer_batch),3)
    average_elapsed_est_timer_batch = np.round(np.average(total_elapsed_est_timer_batch),3)
    print("Average runtime for prediction module: ", average_elapsed_pred_timer_batch, ", where average occupies =", np.round(average_elapsed_pred_timer_batch/average_time_batch*100,3), "% " "relative to total time")
    print("Average runtime for estimation module: ", average_elapsed_est_timer_batch, ", where average occupies =", np.round(average_elapsed_est_timer_batch/average_time_batch*100,3), "% " "relative to total time")

    with open('../runtimes.txt', 'a') as frt:
            frt.write("\n --------------------------------------------------------- \n")
            frt.writelines("\nNumber of beacons used: " + str(num_beacons) + ", Number of simulations ran through: " + str(num_sims) +", Simulation duration (seconds): " + str(N*dt) + "\n")
            frt.writelines("\nEllapsed time for batch:" + str(np.round(elapsed_batch,3)))
            frt.writelines("\nSummed runtime used in prediction module: " + str(np.round(total_elapsed_pred_timer_batch,3)) + "seconds")
            frt.writelines("\nPortion of runtime prediction module occupies: " + str(np.round(total_elapsed_pred_timer_batch/elapsed_batch*100,3)) + "%")
            frt.writelines("\nSummed runtime used in estimation module: " + str(np.round(total_elapsed_est_timer_batch,3)) + "seconds")
            frt.writelines("\nPortion of runtime estimation module occupies: " + str(np.round(total_elapsed_est_timer_batch/elapsed_batch*100,3)) + "%")
            frt.write("\n")
            frt.writelines("\nAverage time for batch elapsed: " +  str(average_time_batch) + "seconds")
            frt.writelines("\nAverage runtime for prediction module: " + str(average_elapsed_pred_timer_batch) + ", where average occupies =" + str(np.round(average_elapsed_pred_timer_batch/average_time_batch*100,3)) + "% " "relative to total time")
            frt.writelines("\nAverage runtime for estimation module: " + str(average_elapsed_est_timer_batch) + ", where average occupies =" + str(np.round(average_elapsed_est_timer_batch/average_time_batch*100,3)) + "% " "relative to total time")
    frt.close()
    
    #if (use_sequential_pseudoranges):
    for i in range(num_sims):
        print("\nUsing sequential pseudoranges without factorization. Run number: ", i+1)  
        # timeit.timeit()
        t_sequential[i] = time.time()
        
        (x_pred,
        x_est,
        P_est,
        GNSSk,
        elapsed_pred_timer_sequential,
        elapsed_est_timer_sequential
        ) = run_sequential_eskf (
                            N, loaded_data,
                            eskf_parameters,
                            x_pred_init, P_pred_init, p_std, 
                            num_beacons,
                            offset =0.0, 
                            use_GNSSaccuracy=False, doGNSS=True,
                            debug=False
                            )
            
        elapsed_sequential[i] = time.time() - t_sequential[i]
        #Summed up time used in prediction steps
        total_elapsed_pred_timer_sequential[i] = np.sum(elapsed_pred_timer_sequential)
        #Summed up time used in estimation steps
        total_elapsed_est_timer_sequential[i] = np.sum(elapsed_est_timer_sequential)

    #Plot the latest run and save figures
    if (N == 10/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '10seq', 'path_seq', x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '10seq', 'path3d_seq', x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '10seq', 'error_pos_sigma_seq')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '10seq', 'error_vel_sigma_seq')
        plot_error_att_sigma(x_est, x_true, P_est, N, '10seq', 'error_att_sigma_seq')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '10seq', 'error_acc_bias_sigma_seq')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '10seq', 'error_rate_bias_sigma_seq')
    
    if (N == 50/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '50seq', 'path_seq', x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '50seq', 'path3d_seq', x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '50seq', 'error_pos_sigma_seq')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '50seq', 'error_vel_sigma_seq')
        plot_error_att_sigma(x_est, x_true, P_est, N, '50seq', 'error_att_sigma_seq')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '50seq', 'error_acc_bias_sigma_seq')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '50seq', 'error_rate_bias_sigma_seq')
    
    if (N == 100/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '100seq', 'path_seq', x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '100seq', 'path3d_seq', x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '100seq', 'error_pos_sigma_seq')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '100seq', 'error_vel_sigma_seq')
        plot_error_att_sigma(x_est, x_true, P_est, N, '100seq', 'error_att_sigma_seq')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '100seq', 'error_acc_bias_sigma_seq')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '100seq', 'error_rate_bias_sigma_seq')
    if (N == 600/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '600seq', 'path_seq', x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '600seq', 'path3d_seq', x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '600seq', 'error_pos_sigma_seq')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '600seq', 'error_vel_sigma_seq')
        plot_error_att_sigma(x_est, x_true, P_est, N, '6006seq', 'error_att_sigma_seq')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '600seq', 'error_acc_bias_sigma_seq')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '600seq', 'error_rate_bias_sigma_seq')
    if (N == 1000/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '1000seq', 'path_seq', x_true)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '1000seq', 'path3d_seq', x_true)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '1000seq', 'error_pos_sigma_seq')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '1000seq', 'error_vel_sigma_seq')
        plot_error_att_sigma(x_est, x_true, P_est, N, '1000seq', 'error_att_sigma_seq')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '1000seq', 'error_acc_bias_sigma_seq')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '1000seq', 'error_rate_bias_sigma_seq')
    
    ## Sequential timings
    
    print("\nEllapsed time for sequential: ", np.round(elapsed_sequential,3))
    print("Summed runtime used in prediction module: ", np.round(total_elapsed_pred_timer_sequential,3), "seconds")
    print("Portion of runtime prediction module occupies: ", np.round(total_elapsed_pred_timer_sequential/elapsed_sequential*100,3),"%")
    print("Summed runtime used in estimation module: ", np.round(total_elapsed_est_timer_sequential,3), "seconds")
    print("Portion of runtime estimation module occupies: ", np.round(total_elapsed_est_timer_sequential/elapsed_sequential*100,3),"%")


    average_time_sequential = np.round(np.average(elapsed_sequential),3)
    print("\nAverage time for sequential elapsed: ", average_time_sequential, "seconds")
    average_elapsed_pred_timer_sequential = np.round(np.average(total_elapsed_pred_timer_sequential),3)
    average_elapsed_est_timer_sequential = np.round(np.average(total_elapsed_est_timer_sequential),3)
    print("Average runtime for prediction module: ", average_elapsed_pred_timer_sequential, ", where average occupies =", np.round(average_elapsed_pred_timer_sequential/average_time_sequential*100,3), "% " "relative to total time")
    print("Average runtime for estimation module: ", average_elapsed_est_timer_sequential, ", where average occupies =", np.round(average_elapsed_est_timer_sequential/average_time_sequential*100,3), "% " "relative to total time")
    
    with open('../runtimes.txt', 'a') as frt:
        frt.write("\n")
        frt.writelines("\nEllapsed time for batch:" + str(np.round(elapsed_sequential,3)))
        frt.writelines("\nSummed runtime used in prediction module: " + str(np.round(total_elapsed_pred_timer_sequential,3)) + "seconds")
        frt.writelines("\nPortion of runtime prediction module occupies: " + str(np.round(total_elapsed_pred_timer_sequential/elapsed_sequential*100,3)) + "%")
        frt.writelines("\nSummed runtime used in estimation module: " + str(np.round(total_elapsed_est_timer_sequential,3)) + "seconds")
        frt.writelines("\nPortion of runtime estimation module occupies: " + str(np.round(total_elapsed_est_timer_sequential/elapsed_sequential*100,3)) + "%")
        frt.write("\n")
        frt.writelines("\nAverage time for batch elapsed: " +  str(average_time_sequential) + "seconds")
        frt.writelines("\nAverage runtime for prediction module: " + str(average_elapsed_pred_timer_sequential) + ", where average occupies =" + str(np.round(average_elapsed_pred_timer_sequential/average_time_sequential*100,3)) + "% " "relative to total time")
        frt.writelines("\nAverage runtime for estimation module: " + str(average_elapsed_est_timer_sequential) + ", where average occupies =" + str(np.round(average_elapsed_est_timer_sequential/average_time_sequential*100,3)) + "% " "relative to total time")
    frt.close()
    
    #if (use_UDU):
    for i in range(num_sims):  
        # timeit.timeit()
        print("Using sequential pseudoranges with UDU Propagation. Run number: ", i+1)
        
        t_UDU[i] = time.time()
        
        (x_pred,
        x_est,
        P_est,
        GNSSk,
        elapsed_pred_timer_UDU,
        elapsed_est_timer_UDU
        ) = run_UDU_eskf (
                            N, loaded_data,
                            use_UDU,
                            eskf_parameters,
                            x_pred_init, P_pred_init, p_std, 
                            num_beacons,
                            offset =0.0, 
                            use_GNSSaccuracy=False, doGNSS=True,
                            debug=False
                            )
        elapsed_UDU[i] = time.time() - t_UDU[i]
        #Summed up time used in prediction steps
        total_elapsed_pred_timer_UDU[i] = np.sum(elapsed_pred_timer_UDU)
        #Summed up time used in estimation steps
        total_elapsed_est_timer_UDU[i] = np.sum(elapsed_est_timer_UDU)
        
    #Plot the latest run and save figures
    if (N == 10/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '10udu', 'path_udu', x_true,)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '10udu', 'path3d_udu', x_true,)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '10udu', 'error_pos_sigma_udu')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '10udu', 'error_vel_sigma_udu')
        plot_error_att_sigma(x_est, x_true, P_est, N, '10udu', 'error_att_sigma_udu')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '10udu', 'error_acc_bias_sigma_udu')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '10udu', 'error_rate_bias_sigma_udu')
    if (N == 50/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '50udu', 'path_udu', x_true,)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '50udu', 'path3d_udu', x_true,)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '50udu', 'error_pos_sigma_udu')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '50udu', 'error_vel_sigma_udu')
        plot_error_att_sigma(x_est, x_true, P_est, N, '50udu', 'error_att_sigma_udu')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '50udu', 'error_acc_bias_sigma_udu')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '50udu', 'error_rate_bias_sigma_udu')
    if (N == 100/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '100udu', 'path_udu', x_true,)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '100udu', 'path3d_udu', x_true,)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '100udu', 'error_pos_sigma_udu')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '100udu', 'error_vel_sigma_udu')
        plot_error_att_sigma(x_est, x_true, P_est, N, '1000udu', 'error_att_sigma_udu')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '1000udu', 'error_acc_bias_sigma_udu')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '1000udu', 'error_rate_bias_sigma_udu')
    if (N == 600/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '600udu', 'path_udu', x_true,)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '600udu', 'path3d_udu', x_true,)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '600udu', 'error_pos_sigma_udu')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '600udu', 'error_vel_sigma_udu')
        plot_error_att_sigma(x_est, x_true, P_est, N, '600udu', 'error_att_sigma_udu')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '600udu', 'error_acc_bias_sigma_udu')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '600udu', 'error_rate_bias_sigma_udu')
    if (N == 1000/dt):
        plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '1000udu', 'path_udu', x_true,)
        plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, '1000udu', 'path3d_udu', x_true,)
        plot_error_pos_sigma(x_est, x_true, P_est, N, '1000udu', 'error_pos_sigma_udu')
        plot_error_vel_sigma(x_est, x_true, P_est, N, '1000udu', 'error_vel_sigma_udu')
        plot_error_att_sigma(x_est, x_true, P_est, N, '1000udu', 'error_att_sigma_udu')
        plot_error_acc_bias_sigma(x_est, x_true, P_est, N, '1000udu', 'error_acc_bias_sigma_udu')  
        plot_error_rate_bias_sigma(x_est, x_true, P_est, N, '1000udu', 'error_rate_bias_sigma_udu')
    ## UDU timings

    print("\nEllapsed time for UDU: ",  np.round(elapsed_UDU,3))
    print("Summed runtime used in UDU prediction module: ",  np.round(total_elapsed_pred_timer_UDU,3), "seconds")
    print("Portion of runtime prediction module in UDU occupies: ",  np.round(total_elapsed_pred_timer_UDU/elapsed_UDU,3),"%")
    print("Summed runtime used in UDU estimation module: ",  np.round(total_elapsed_est_timer_UDU,3), "seconds")
    print("Portion of runtime estimation module in UDU ccupies: ",  np.round(total_elapsed_est_timer_UDU/elapsed_UDU,3),"%")

    average_time_UDU = np.round(np.average(elapsed_UDU),3)
    print("\nAverage time for UDU elapsed: ", np.round(average_time_UDU,3), "seconds")
    average_elapsed_pred_timer_UDU = np.round(np.average(total_elapsed_pred_timer_UDU),3)
    average_elapsed_est_timer_UDU = np.round(np.average(total_elapsed_est_timer_UDU),3)
    print("Average runtime for prediction module in UDU: ", average_elapsed_pred_timer_UDU, ", where average occupies =",  np.round(average_elapsed_pred_timer_UDU/average_time_UDU*100,3), "% " "relative to total time")
    print("Average runtime for estimation module in UDU: ", average_elapsed_est_timer_UDU, ", where average occupies =",  np.round(average_elapsed_est_timer_UDU/average_time_UDU*100,3), "% " "relative to total time")


    with open('../runtimes.txt', 'a') as frt:
        frt.write("\n")
        frt.writelines("\nEllapsed time for batch:" + str(np.round(elapsed_UDU,3)))
        frt.writelines("\nSummed runtime used in prediction module: " + str(np.round(total_elapsed_pred_timer_UDU,3)) + "seconds")
        frt.writelines("\nPortion of runtime prediction module occupies: " + str(np.round(total_elapsed_pred_timer_UDU/elapsed_UDU*100,3)) + "%")
        frt.writelines("\nSummed runtime used in estimation module: " + str(np.round(total_elapsed_est_timer_UDU,3)) + "seconds")
        frt.writelines("\nPortion of runtime estimation module occupies: " + str(np.round(total_elapsed_est_timer_UDU/elapsed_UDU*100,3)) + "%")
        frt.write("\n")
        frt.writelines("\nAverage time for batch elapsed: " +  str(elapsed_UDU) + "seconds")
        frt.writelines("\nAverage runtime for prediction module: " + str(average_elapsed_pred_timer_UDU) + ", where average occupies =" + str(np.round(average_elapsed_pred_timer_UDU/average_time_UDU*100,3)) + "% " "relative to total time")
        frt.writelines("\nAverage runtime for estimation module: " + str(average_elapsed_est_timer_UDU) + ", where average occupies =" + str(np.round(average_elapsed_est_timer_UDU/average_time_UDU*100,3)) + "% " "relative to total time")
    frt.close()
    

    ## Relative relations
    if (use_batch_pseudoranges & use_sequential_pseudoranges):
        print("\nAverage Relative speedup of batch vs sequential: ",  np.round((average_time_batch - average_time_sequential)/average_time_batch*100,3),"%")
        print("Average Relative speedup of pred module in batch vs sequential: ",  np.round((average_elapsed_pred_timer_batch - average_elapsed_pred_timer_sequential)/average_elapsed_pred_timer_batch*100,3), "%")
        print("Average Relative speedup of est module in batch vs sequential: ",  np.round((average_elapsed_est_timer_batch - average_elapsed_est_timer_sequential)/average_elapsed_est_timer_batch*100,3), "%")
        
        with open('../runtimes.txt', 'a') as frt:
            frt.write("\n")
            frt.writelines("\n\nAverage Relative speedup of batch vs sequential: " +  str(np.round((average_time_batch - average_time_sequential)/average_time_batch*100,3)) + "%")
            frt.writelines("\nAverage Relative speedup of pred module in batch vs sequential: " +  str(np.round((average_elapsed_pred_timer_batch - average_elapsed_pred_timer_sequential)/average_elapsed_pred_timer_batch*100,3)) + "%")
            frt.writelines("\nAverage Relative speedup of est module in batch vs sequential: " +  str(np.round((average_elapsed_est_timer_batch - average_elapsed_est_timer_sequential)/average_elapsed_est_timer_batch*100,3)) + "%")
        frt.close()

    if (use_batch_pseudoranges & use_UDU):
        print("\nAverage Relative speedup of batch vs UDU-sequential: ",  np.round((average_time_batch - average_time_UDU)/average_time_batch*100,3),"%")
        print("Average Relative speedup of pred module in batch vs UDU-sequential: ",  np.round((average_elapsed_pred_timer_batch - average_elapsed_pred_timer_UDU)/average_elapsed_pred_timer_batch*100,3), "%")
        print("Average Relative speedup of est module in batch vs UDU-sequential: ",  np.round((average_elapsed_est_timer_batch - average_elapsed_est_timer_UDU)/average_elapsed_est_timer_batch*100,3), "%")
        
        with open('../runtimes.txt', 'a') as frt:
            frt.write("\n")
            frt.writelines("\n\nAverage Relative speedup of batch vs UDU-sequential: " +  str(np.round((average_time_batch - average_time_UDU)/average_time_batch*100,3)) + "%")
            frt.writelines("\nAverage Relative speedup of pred module in batch vs UDU-sequential: " +  str(np.round((average_elapsed_pred_timer_batch - average_elapsed_pred_timer_UDU)/average_elapsed_pred_timer_batch*100,3)) + "%")
            frt.writelines("\nAverage Relative speedup of est module in batch vs UDU-sequential: " +  str(np.round((average_elapsed_est_timer_batch - average_elapsed_est_timer_UDU)/average_elapsed_est_timer_batch*100,3)) + "%")
        frt.close()

    if (use_UDU & use_sequential_pseudoranges):
        print("\nAverage Relative speedup of sequential vs UDU-sequential: ",  np.round((average_time_sequential - average_time_UDU)/average_time_sequential*100,3),"%")
        print("Average Relative speedup of pred module in sequential vs UDU-sequential: ",  np.round((average_elapsed_pred_timer_sequential - average_elapsed_pred_timer_UDU)/average_elapsed_pred_timer_sequential*100,3), "%")
        print("Average Relative speedup of est module in  sequential vs UDU-sequential: ",  np.round((average_elapsed_est_timer_sequential - average_elapsed_est_timer_UDU)/average_elapsed_est_timer_sequential*100,3), "%")
        with open('../runtimes.txt', 'a') as frt:
            frt.write("\n")
            frt.writelines("\n\nAverage Relative speedup of sequential vs UDU-sequential: " +  str(np.round((average_time_sequential - average_time_UDU)/average_time_sequential*100,3)) + "%")
            frt.writelines("\nAverage Relative speedup of pred module in sequential vs UDU-sequential: " +  str(np.round((average_elapsed_pred_timer_sequential - average_elapsed_pred_timer_UDU)/average_elapsed_pred_timer_sequential*100,3)) + "%")
            frt.writelines("\nAverage Relative speedup of est module in sequential vs UDU-sequential: " +  str(np.round((average_elapsed_est_timer_sequential - average_elapsed_est_timer_UDU)/average_elapsed_est_timer_sequential*100,3)) + "%")
        frt.close()

        
    if (use_batch_pseudoranges & use_sequential_pseudoranges & use_UDU):
        if (N ==10/dt):
            plot_timing_scatter('timing10','1Total_elapsed_box','total',elapsed_batch,elapsed_sequential,elapsed_UDU)
            plot_timing_scatter('timing10','1Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential,total_elapsed_pred_timer_UDU)
            plot_timing_scatter('timing10','1Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential,total_elapsed_est_timer_UDU)
        if (N ==50/dt):
            plot_timing_scatter('timing50','1Total_elapsed_box','total',elapsed_batch,elapsed_sequential,elapsed_UDU)
            plot_timing_scatter('timing50','1Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential,total_elapsed_pred_timer_UDU)
            plot_timing_scatter('timing50','1Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential,total_elapsed_est_timer_UDU)
            
        if (N ==100/dt):
            plot_timing_scatter('timing100','1Total_elapsed_box','total',elapsed_batch,elapsed_sequential,elapsed_UDU)
            plot_timing_scatter('timing100','1Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential,total_elapsed_pred_timer_UDU)
            plot_timing_scatter('timing100','1Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential,total_elapsed_est_timer_UDU)                
            
        if (N ==600/dt):
            plot_timing_scatter('timing600','1Total_elapsed_box','total',elapsed_batch,elapsed_sequential,elapsed_UDU)
            plot_timing_scatter('timing600','1Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential,total_elapsed_pred_timer_UDU)
            plot_timing_scatter('timing600','1Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential,total_elapsed_est_timer_UDU)                
            
        if (N ==1000/dt):
            plot_timing_scatter('timing1000','1Total_elapsed_box','total',elapsed_batch,elapsed_sequential,elapsed_UDU)
            plot_timing_scatter('timing1000','1Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential,total_elapsed_pred_timer_UDU)
            plot_timing_scatter('timing1000','1Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential,total_elapsed_est_timer_UDU)                                

    if (use_batch_pseudoranges & use_sequential_pseudoranges):
        if (N ==10/dt):
            plot_timing_scatter2('timing10','2Total_elapsed_box','total',elapsed_batch,elapsed_sequential)
            plot_timing_scatter2('timing10','2Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential)
            plot_timing_scatter2('timing10','2Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential)   
        if (N ==50/dt):
            plot_timing_scatter2('timing50','2Total_elapsed_box','total',elapsed_batch,elapsed_sequential)
            plot_timing_scatter2('timing50','2Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential)
            plot_timing_scatter2('timing50','2Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential)                                
        if (N ==100/dt):
            plot_timing_scatter2('timing100','2Total_elapsed_box','total',elapsed_batch,elapsed_sequential)
            plot_timing_scatter2('timing100','2Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential)
            plot_timing_scatter2('timing100','2Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential)                
        if (N ==600/dt):
            plot_timing_scatter2('timing600','2Total_elapsed_box','total',elapsed_batch,elapsed_sequential)
            plot_timing_scatter2('timing600','2Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential)
            plot_timing_scatter2('timing600','2Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential)                
        if (N ==1000/dt):
            plot_timing_scatter2('timing1000','2Total_elapsed_box','total',elapsed_batch,elapsed_sequential)
            plot_timing_scatter2('timing1000','2Pred_elapsed_box','time update module',total_elapsed_pred_timer_batch,total_elapsed_pred_timer_sequential)
            plot_timing_scatter2('timing1000','2Est_elapsed_box','measurement update module',total_elapsed_est_timer_batch,total_elapsed_est_timer_sequential)                
            



