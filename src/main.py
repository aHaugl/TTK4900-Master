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
from eskf_UDU import ESKF_UDU
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


# %% Run estimation for
#Number of seconds of simulation to run. len(timeIMU) decides max
# N: int = int(6000/dt)
N: int = int(50/dt) 
# N: int = int(30000) 
# N: int = int(90000)
# N: int = len(timeIMU)
offset = 0
doGNSS: bool = True
rtol = 1e-05
atol = 1e-08

# %% Plots and stuff                           

# plt.close("all")
t = np.linspace(0,dt * (N-1), N)
# plot_path(t, N, pos_t, pos_t)



# %% Run eskf_runner  
"""
Running the simulation
"""
beacon_location: np.ndarray = loaded_data["beacon_location"]

use_batch_pseudoranges: bool = True
# # use_batch_pseudoranges: bool = False
use_sequential_pseudoranges: bool = True
# use_sequential_pseudoranges: bool = False
use_UDU: bool = False



num_beacons = len(beacon_location)
num_sims = 10

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


# %%
print("Number of beacons used: ", num_beacons)
print("Number of simulations ran through", num_sims)
print("Simulation duration (seconds): ", N*dt) 


# %%
if (use_batch_pseudoranges):
    
    for i in range(num_sims):  
        # timeit.timeit()
        print("Using batch pseudoranges without factorization. Run number: ", i+1)
        
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
                            rtol,
                            atol,
                            Use_UDU,
                            Use_QR,
                            Use_LU,
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


    plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
    plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)

    # print("Ellapsed time for batch: ", elapsed_batch)
    # print("Summed runtime used in prediction module: ", total_elapsed_pred_timer_batch, "seconds")
    # print("Portion of runtime prediction module occupies: ", total_elapsed_pred_timer_batch/elapsed_batch, "%")
    # print("Summed runtime used in estimation module: ", total_elapsed_est_timer_batch, "seconds")
    # print("Portion of runtime estimation module occupies: ", total_elapsed_est_timer_batch/elapsed_batch, "%")
  

    # average_time_batch = np.average(elapsed_batch)
    # print("Average time for batch elapsed: ", average_time_batch, "seconds")
    # average_elapsed_pred_timer_batch = np.average(total_elapsed_pred_timer_batch)
    # average_elapsed_est_timer_batch = np.average(total_elapsed_est_timer_batch)
    # print("Average runtime for prediction module: ", average_elapsed_pred_timer_batch, ", where average occupies =", average_elapsed_pred_timer_batch/average_time_batch*100, "% " "relative to total time")
    # print("Average runtime for estimation module: ", average_elapsed_est_timer_batch, ", where average occupies =", average_elapsed_est_timer_batch/average_time_batch*100, "% " "relative to total time")

if (use_sequential_pseudoranges):
    for i in range(num_sims):
        print("Using sequential pseudoranges without factorization. Run number: ", i+1)  
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
                            rtol,
                            atol,
                            Use_UDU,
                            Use_QR,
                            Use_LU,
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

    plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
    plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
    
if (use_UDU):
    
    for i in range(num_sims):  
        # timeit.timeit()
        print("Using batch pseudoranges without factorization. Run number: ", i+1)
        
        t_UDU[i] = time.time()
        
        (x_pred,
        x_est,
        P_est,
        GNSSk,
        est_timer,
        pred_timer,
        elapsed_pred_timer_UDU,
        elapsed_est_timer_UDU
        ) = run_batch_eskf (
                            N, loaded_data,
                            rtol,
                            atol,
                            Use_UDU,
                            Use_QR,
                            Use_LU,
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

## Batch timings
    print("Ellapsed time for batch: ", elapsed_batch)
    print("Summed runtime used in prediction module: ", total_elapsed_pred_timer_batch, "seconds")
    print("Portion of runtime prediction module occupies: ", total_elapsed_pred_timer_batch/elapsed_batch, "%")
    print("Summed runtime used in estimation module: ", total_elapsed_est_timer_batch, "seconds")
    print("Portion of runtime estimation module occupies: ", total_elapsed_est_timer_batch/elapsed_batch, "%")
  

    average_time_batch = np.average(elapsed_batch)
    print("Average time for batch elapsed: ", average_time_batch, "seconds")
    average_elapsed_pred_timer_batch = np.average(total_elapsed_pred_timer_batch)
    average_elapsed_est_timer_batch = np.average(total_elapsed_est_timer_batch)
    print("Average runtime for prediction module: ", average_elapsed_pred_timer_batch, ", where average occupies =", average_elapsed_pred_timer_batch/average_time_batch*100, "% " "relative to total time")
    print("Average runtime for estimation module: ", average_elapsed_est_timer_batch, ", where average occupies =", average_elapsed_est_timer_batch/average_time_batch*100, "% " "relative to total time")

## Iterative timings
    print("Ellapsed time for sequential: ", elapsed_sequential)
    print("Summed runtime used in prediction module: ", total_elapsed_pred_timer_sequential, "seconds")
    print("Portion of runtime prediction module occupies: ", total_elapsed_pred_timer_sequential/elapsed_sequential,"%")
    print("Summed runtime used in estimation module: ", total_elapsed_est_timer_sequential, "seconds")
    print("Portion of runtime estimation module occupies: ", total_elapsed_est_timer_sequential/elapsed_sequential,"%")
  

    average_time_sequential = np.average(elapsed_sequential)
    print("Average time for sequential elapsed: ", average_time_sequential, "seconds")
    average_elapsed_pred_timer_sequential = np.average(total_elapsed_pred_timer_sequential)
    average_elapsed_est_timer_sequential = np.average(total_elapsed_est_timer_sequential)
    print("Average runtime for prediction module: ", average_elapsed_pred_timer_sequential, ", where average occupies =", average_elapsed_pred_timer_sequential/average_time_sequential*100, "% " "relative to total time")
    print("Average runtime for estimation module: ", average_elapsed_est_timer_sequential, ", where average occupies =", average_elapsed_est_timer_sequential/average_time_sequential*100, "% " "relative to total time")

    ## UDU timings
    print("Ellapsed time for UDU: ", elapsed_UDU)
    print("Summed runtime used in UDU prediction module: ", total_elapsed_pred_timer_UDU, "seconds")
    print("Portion of runtime prediction module in UDU occupies: ", total_elapsed_pred_timer_UDU/elapsed_UDU,"%")
    print("Summed runtime used in UDU estimation module: ", total_elapsed_est_timer_UDU, "seconds")
    print("Portion of runtime estimation module in UDU ccupies: ", total_elapsed_est_timer_UDU/elapsed_UDU,"%")
  

    average_time_UDU = np.average(elapsed_UDU)
    print("Average time for UDU elapsed: ", average_time_UDU, "seconds")
    average_elapsed_pred_timer_UDU = np.average(total_elapsed_pred_timer_UDU)
    average_elapsed_est_timer_UDU = np.average(total_elapsed_est_timer_UDU)
    print("Average runtime for prediction module in UDU: ", average_elapsed_pred_timer_UDU, ", where average occupies =", average_elapsed_pred_timer_UDU/average_time_UDU*100, "% " "relative to total time")
    print("Average runtime for estimation module in UDU: ", average_elapsed_est_timer_UDU, ", where average occupies =", average_elapsed_est_timer_UDU/average_time_UDU*100, "% " "relative to total time")

## Relative relations
print("Average Relative speedup of sequential vs batch: ", (average_time_batch - average_time_sequential)/average_time_batch*100,"%")
print("Average Relative speedup of pred module in sequential vs batch: ", (average_elapsed_pred_timer_batch - average_elapsed_pred_timer_sequential)/average_elapsed_pred_timer_batch*100,"%")
print("Average Relative speedup of est module in  sequential vs batch: ", (average_elapsed_est_timer_batch - average_elapsed_est_timer_sequential)/average_elapsed_est_timer_batch*100,"%")

print("Average Relative speedup of UDU vs batch: ", (average_time_batch - average_time_UDU)/average_time_batch*100,"%")
print("Average Relative speedup of pred module in sequential vs batch: ", (average_elapsed_pred_timer_batch - average_elapsed_pred_timer_UDU)/average_elapsed_pred_timer_batch*100,"%")
print("Average Relative speedup of est module in  sequential vs batch: ", (average_elapsed_est_timer_batch - average_elapsed_est_timer_UDU)/average_elapsed_est_timer_batch*100,"%")


print("Average Relative speedup of UDU vs sequential: ", (average_time_sequential - average_time_UDU)/average_time_sequential*100,"%")
print("Average Relative speedup of pred module in sequential vs batch: ", (average_elapsed_pred_timer_sequential - average_elapsed_pred_timer_UDU)/average_elapsed_pred_timer_sequential*100,"%")
print("Average Relative speedup of est module in  sequential vs batch: ", (average_elapsed_est_timer_sequential - average_elapsed_est_timer_UDU)/average_elapsed_est_timer_sequential*100,"%")
# %% Plots and stuff                           

# # plt.close("all")
# t = np.linspace(0,dt * (N-1), N)
# # plot_path(t, N, pos_t, pos_t)
# tGNSS = loaded_data["timeGNSS"].T
# z_GNSS = loaded_data["z_GNSS"].T

# z_acc_vector = loaded_data["z_acc"].T
# acc_t = loaded_data["acc_t"].T
# omega_t = loaded_data["omega_t"].T
# z_gyro_vector = loaded_data["z_gyro"].T


# %% Estimation plots

# plot_pos(t, N, x_est, tGNSS, GNSSk, z_GNSS, x_true)
# plot_vel(t, N, x_est, x_true)
# plot_acc(t, N, z_acc_vector, acc_t)
# # %%
# # plot_gyro(t, N, z_gyro_vector, omega_t)
# # %%
# plot_angle(t, N, x_est, x_true)
# # 
# # plot_estimate(t, N, x_est)

# plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
# # # %%
# plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
# # # %%

# state_error_plots(t, N, x_est, x_true, delta_x)

# # %% Validation plots
# plot_error_pos_sigma(x_est, x_true, P_est, N)
# plot_error_vel_sigma(x_est, x_true, P_est, N)
# plot_error_att_sigma(x_est, x_true, P_est, N)
# plot_error_acc_bias_sigma(x_est, x_true, P_est, N)
# plot_error_rate_bias_sigma(x_est, x_true, P_est, N)

# # error_distance_plot(t, N, dt, GNSSk, x_true, delta_x, z_GNSS)
# # %% 
# plot_NIS(NIS)
# plot_NEES(t, N, dt,
#           NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias, NEES_gyrobias,
#             confprob=0.95)

# %%
