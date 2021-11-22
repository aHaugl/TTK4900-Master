# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:39:06 2021

@author: Andreas
"""
import numpy as np
from tqdm import trange
from tqdm import tqdm_notebook
import timeit
import time
# from timer import * 

from eskf_batch import (
# from eskf import (    
    ESKF_batch,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from eskf_sequential import (
# from eskf import (    
    ESKF_sequential,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from eskf_UDU import (
# from eskf import (    
    ESKF_udu,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from IMU import z_acc, z_gyro


def run_batch_eskf(
                N, loaded_data,
                eskf_parameters,
                x_pred_init, P_pred_init, p_std, 
                num_beacons,
                offset =0.0, 
                use_GNSSaccuracy=False, doGNSS=True,
                debug=False
                ):
    """
    Description:
        Unravels, initializes and runs error state kalman filter 
        for parameters set in .mat file and main

    Parameters
    ----------
    N :  Number of steps to run
    loaded_data : Loaded data matrix
    eskf_parameters : 
    x_pred_init : State prediction initialization params
    P_pred_init_list : Covariance init params
    use_GNSSaccuracy : TYPE, optional
        DESCRIPTION. The default is False.
    doGNSS : TYPE, optional
        DESCRIPTION. The default is False.
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    offset : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    result : List of returns from ESKF

    """
    
    # %% Read loaded data
    if "x_true" in loaded_data:
        x_true = loaded_data["x_true"].T
    else:
        x_true = None
    z_acc = loaded_data["z_acc"].T
    z_gyro = loaded_data["z_gyro"].T
    z_GNSS = loaded_data["z_GNSS"].T
#    acc_t = loaded_data["acc_t"].T
    
    if use_GNSSaccuracy:
        print("Using GNSS_accuracy")
        GNSSaccuracy = loaded_data['GNSSaccuracy'].T
    else:
        print("Not using GNSS_accuracy")
        GNSSaccuracy = None
    
    # S_a: np.ndarray = np.eye(3)
    # S_g: np.ndarray = np.eye(3)
    
    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]
    
    # %% Beacon stuff
    # The matrix has 15 beacons implemented
    beacon_location: np.ndarray = loaded_data["beacon_location"]
    #To select x amount of beacons, use beacon_location = beacon_location[:x], x =[0,15]
    beacon_location = beacon_location[:num_beacons]
    R_beacons = np.zeros((num_beacons,num_beacons))
    np.fill_diagonal(R_beacons, p_std**2)
    # np.fill_diagonal(R_beacons,np.array([1,1,1.5])**2)
    
    # %% 
    lever_arm = loaded_data["leverarm"].ravel()
    
    timeGNSS = loaded_data["timeGNSS"].ravel()
    timeIMU = loaded_data["timeIMU"].ravel()
    Ts_IMU = [0, *np.diff(timeIMU)]

    
    steps = len(z_acc)
    gnss_steps = len(z_GNSS)
        
    # %% Initialize state predictions, estimates and NEES   
    x_pred: np.ndarray = np.zeros((steps, 16))
    x_pred[0] = x_pred_init

    P_pred = np.zeros((N, 15, 15))
    P_pred[0] = P_pred_init
    
        #Initialize the kalman filter
    print("Initializing kalman filter")
    eskf = ESKF_batch(
        *eskf_parameters,
        S_a=S_a,  # set the accelerometer correction matrix
        S_g=S_g,  # set the gyro correction matrix,
        debug=debug

    )
    R_GNSS = np.diag(p_std ** 2)
    
    x_est: np.ndarray  = np.zeros((N, 16))
    P_est = np.zeros((N, 15, 15))
  
     # keep track of current step in GNSS measurements
    offset += timeIMU[0]
    GNSSk_init = np.searchsorted(timeGNSS, offset)
    GNSSk = GNSSk_init
    offset_idx = np.searchsorted(timeIMU, offset)
    timeIMU = timeIMU[offset_idx:]
    z_acc = z_acc[offset_idx:]
    z_gyro = z_gyro[offset_idx:]
    Ts_IMU = Ts_IMU[offset_idx:]
    k = 0

    ## Timers for benchmarking
    pred_timer = np.zeros(N)
    elapsed_pred_timer = np.zeros(N)
    est_timer = np.zeros(int(N*0.01))
    elapsed_est_timer = np.zeros(int(N*0.01))

    # %% 
    # print("Starting timer")
    # tic()
    print("Running filter on simulation model")
    for k in trange(N):
        if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
            if use_GNSSaccuracy:
                R_GNSS_scaled = R_GNSS * GNSSaccuracy[GNSSk]
            else:
                R_GNSS_scaled = R_GNSS

            # Only counting when estimation does something else than prediction, i.e skipping 
            # the else steps 
            est_timer[GNSSk] = time.time()
            x_est[k], P_est[k] = eskf.update_GNSS_position(
                                        x_pred[k],
                                        P_pred[k],
                                        z_GNSS[GNSSk],
                                        R_GNSS_scaled,
                                        R_beacons,
                                        beacon_location,
                                        lever_arm
                                        )
            

            #Ranges
            assert np.all(np.isfinite(P_est[k])
                          ), f"Not finite P_pred at index {k}"
            elapsed_est_timer[GNSSk] = time.time() - est_timer[GNSSk]
            GNSSk += 1
            
        else:
            # No updates, est = pred
            x_est[k] = x_pred[k]
            P_est[k] = P_pred[k]

            
        if k < N - 1:
            pred_timer[k] = time.time()
            x_pred[k+1], P_pred[k+1] = eskf.predict(
                                            x_est[k],
                                            P_est[k],
                                            # acc_t[k],
                                            z_acc[k], #Denne er k pga måten dataen er laget på (?)
                                            z_gyro[k+1],
                                            Ts_IMU[k]
                                             )
            elapsed_pred_timer[k] = time.time() - pred_timer[k]

    result_batch = (
            x_pred,
            x_est,
            P_est,
            GNSSk,
            est_timer,
            pred_timer,
            elapsed_pred_timer,
            elapsed_est_timer  
              )


    return result_batch


    
def run_sequential_eskf(
                    N, loaded_data,
                    eskf_parameters,
                    x_pred_init, P_pred_init, p_std, 
                    num_beacons,
                    offset =0.0, 
                    use_GNSSaccuracy=False, doGNSS=True,
                    debug=False
                    ):
    """
    Description:
        Unravels, initializes and runs error state kalman filter 
        for parameters set in .mat file and main

    Parameters
    ----------
    N :  Number of steps to run
    loaded_data : Loaded data matrix
    eskf_parameters : 
    x_pred_init : State prediction initialization params
    P_pred_init_list : Covariance init params
    use_GNSSaccuracy : TYPE, optional
        DESCRIPTION. The default is False.
    doGNSS : TYPE, optional
        DESCRIPTION. The default is False.
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    offset : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    result : List of returns from ESKF

    """
    
    # %% Read loaded data
    if "x_true" in loaded_data:
        x_true = loaded_data["x_true"].T
    else:
        x_true = None
    z_acc = loaded_data["z_acc"].T
    z_gyro = loaded_data["z_gyro"].T
    z_GNSS = loaded_data["z_GNSS"].T
#    acc_t = loaded_data["acc_t"].T
    
    if use_GNSSaccuracy:
        print("Using GNSS_accuracy")
        GNSSaccuracy = loaded_data['GNSSaccuracy'].T
    else:
        print("Not using GNSS_accuracy")
        GNSSaccuracy = None
    
    # S_a: np.ndarray = np.eye(3)
    # S_g: np.ndarray = np.eye(3)
    
    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]
    
    # %% Beacon stuff
    # The matrix has 15 beacons implemented
    beacon_location: np.ndarray = loaded_data["beacon_location"]
    #To select x amount of beacons, use beacon_location = beacon_location[:x], x =[0,15]
    beacon_location = beacon_location[:num_beacons]
    R_beacons = np.zeros((num_beacons,num_beacons))
    np.fill_diagonal(R_beacons, p_std**2)
    # np.fill_diagonal(R_beacons,np.array([1,1,1.5])**2)
    
    # %% 
    lever_arm = loaded_data["leverarm"].ravel()
    
    timeGNSS = loaded_data["timeGNSS"].ravel()
    timeIMU = loaded_data["timeIMU"].ravel()
    Ts_IMU = [0, *np.diff(timeIMU)]
    
    steps = len(z_acc)
    gnss_steps = len(z_GNSS)
        
    # %% Initialize state predictions, estimates and NEES   
    x_pred: np.ndarray = np.zeros((steps, 16))
    x_pred[0] = x_pred_init

    P_pred = np.zeros((N, 15, 15))
    P_pred[0] = P_pred_init
    
        #Initialize the kalman filter
    print("Initializing kalman filter")
    eskf = ESKF_sequential(
        *eskf_parameters,
        S_a=S_a,  # set the accelerometer correction matrix
        S_g=S_g,  # set the gyro correction matrix,
        debug=debug
    )
    R_GNSS = np.diag(p_std ** 2)
    
    x_est: np.ndarray  = np.zeros((N, 16))
    P_est = np.zeros((N, 15, 15))
  
     # keep track of current step in GNSS measurements
    offset += timeIMU[0]
    GNSSk_init = np.searchsorted(timeGNSS, offset)
    GNSSk = GNSSk_init
    offset_idx = np.searchsorted(timeIMU, offset)
    timeIMU = timeIMU[offset_idx:]
    z_acc = z_acc[offset_idx:]
    z_gyro = z_gyro[offset_idx:]
    Ts_IMU = Ts_IMU[offset_idx:]
    k = 0

    ## Timers for benchmarking
    pred_timer = np.zeros(N)
    elapsed_pred_timer = np.zeros(N)
    est_timer = np.zeros(int(N*0.01))
    elapsed_est_timer = np.zeros(int(N*0.01))

    # %% 
    # print("Starting timer")
    # tic()
    print("Running filter on simulation model")
    for k in trange(N):
        if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
            if use_GNSSaccuracy:
                R_GNSS_scaled = R_GNSS * GNSSaccuracy[GNSSk]
            else:
                R_GNSS_scaled = R_GNSS


            # %% Test if range and LOS_matrix works
            est_timer[GNSSk] = time.time()
            x_est[k], P_est[k] = eskf.update_GNSS_position(
                                        x_pred[k],
                                        P_pred[k],
                                        z_GNSS[GNSSk],
                                        R_GNSS_scaled,
                                        R_beacons,
                                        beacon_location,
                                        lever_arm
                                        )

            #Ranges
            assert np.all(np.isfinite(P_est[k])
                          ), f"Not finite P_pred at index {k}"
            elapsed_est_timer[GNSSk] = time.time() - est_timer[GNSSk]
            GNSSk += 1
        else:
            # No updates, est = pred
            x_est[k] = x_pred[k]
            P_est[k] = P_pred[k]

            
        if k < N - 1:
            pred_timer[k] = time.time()
            x_pred[k+1], P_pred[k+1] = eskf.predict(
                                            x_est[k],
                                            P_est[k],
                                            # acc_t[k],
                                            z_acc[k], #Denne er k pga måten dataen er laget på (?)
                                            z_gyro[k+1],
                                            Ts_IMU[k]
                                             )
            elapsed_pred_timer[k] = time.time() - pred_timer[k]
    result_sequential = (
            x_pred,
            x_est,
            P_est,
            GNSSk,
            elapsed_pred_timer,
            elapsed_est_timer       
              )


    return result_sequential

def run_UDU_eskf(
                    N, loaded_data,
                    use_UDU,
                    eskf_parameters,
                    x_pred_init, P_pred_init, p_std, 
                    num_beacons,
                    offset =0.0, 
                    use_GNSSaccuracy=False, doGNSS=True,
                    debug=False
                    ):
    """
    Description:
        Unravels, initializes and runs error state kalman filter 
        for parameters set in .mat file and main

    Parameters
    ----------
    N :  Number of steps to run
    loaded_data : Loaded data matrix
    eskf_parameters : 
    x_pred_init : State prediction initialization params
    P_pred_init_list : Covariance init params
    use_GNSSaccuracy : TYPE, optional
        DESCRIPTION. The default is False.
    doGNSS : TYPE, optional
        DESCRIPTION. The default is False.
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    offset : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    result : List of returns from ESKF

    """
    
    # %% Read loaded data
    if "x_true" in loaded_data:
        x_true = loaded_data["x_true"].T
    else:
        x_true = None
    z_acc = loaded_data["z_acc"].T
    z_gyro = loaded_data["z_gyro"].T
    z_GNSS = loaded_data["z_GNSS"].T
#    acc_t = loaded_data["acc_t"].T
    
    if use_GNSSaccuracy:
        print("Using GNSS_accuracy")
        GNSSaccuracy = loaded_data['GNSSaccuracy'].T
    else:
        print("Not using GNSS_accuracy")
        GNSSaccuracy = None
    
    # S_a: np.ndarray = np.eye(3)
    # S_g: np.ndarray = np.eye(3)
    
    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]
    
    # %% Beacon stuff
    # The matrix has 15 beacons implemented
    beacon_location: np.ndarray = loaded_data["beacon_location"]
    #To select x amount of beacons, use beacon_location = beacon_location[:x], x =[0,15]
    beacon_location = beacon_location[:num_beacons]
    R_beacons = np.zeros((num_beacons,num_beacons))
    np.fill_diagonal(R_beacons, p_std**2)
    # np.fill_diagonal(R_beacons,np.array([1,1,1.5])**2)
    
    # %% 
    lever_arm = loaded_data["leverarm"].ravel()
    
    timeGNSS = loaded_data["timeGNSS"].ravel()
    timeIMU = loaded_data["timeIMU"].ravel()
    Ts_IMU = [0, *np.diff(timeIMU)]
    
    steps = len(z_acc)
    gnss_steps = len(z_GNSS)
        
    # %% Initialize state predictions, estimates and NEES   
    x_pred: np.ndarray = np.zeros((steps, 16))
    x_pred[0] = x_pred_init

    P_pred = np.zeros((N, 15, 15))
    P_pred[0] = P_pred_init
    
        #Initialize the kalman filter
    print("Initializing kalman filter")
    eskf = ESKF_udu(
        *eskf_parameters,
        S_a=S_a,  # set the accelerometer correction matrix
        S_g=S_g,  # set the gyro correction matrix,
        debug=debug
    )
    R_GNSS = np.diag(p_std ** 2)
    
    x_est: np.ndarray  = np.zeros((N, 16))
    P_est = np.zeros((N, 15, 15))
  
     # keep track of current step in GNSS measurements
    offset += timeIMU[0]
    GNSSk_init = np.searchsorted(timeGNSS, offset)
    GNSSk = GNSSk_init
    offset_idx = np.searchsorted(timeIMU, offset)
    timeIMU = timeIMU[offset_idx:]
    z_acc = z_acc[offset_idx:]
    z_gyro = z_gyro[offset_idx:]
    Ts_IMU = Ts_IMU[offset_idx:]
    k = 0

    ## Timers for benchmarking
    pred_timer = np.zeros(N)
    elapsed_pred_timer = np.zeros(N)
    est_timer = np.zeros(int(N*0.01))
    elapsed_est_timer = np.zeros(int(N*0.01))

    # %% 
    # print("Starting timer")
    # tic()
    print("Running filter on simulation model")
    for k in trange(N):
        if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
            if use_GNSSaccuracy:
                R_GNSS_scaled = R_GNSS * GNSSaccuracy[GNSSk]
            else:
                R_GNSS_scaled = R_GNSS


            # %% Test if range and LOS_matrix works
            est_timer[GNSSk] = time.time()
            x_est[k], P_est[k] = eskf.update_GNSS_position(
                                        use_UDU,
                                        x_pred[k],
                                        P_pred[k],
                                        z_GNSS[GNSSk],
                                        R_GNSS_scaled,
                                        R_beacons,
                                        beacon_location,
                                        lever_arm
                                        )

            #Ranges
            assert np.all(np.isfinite(P_est[k])
                          ), f"Not finite P_pred at index {k}"
            elapsed_est_timer[GNSSk] = time.time() - est_timer[GNSSk]
            GNSSk += 1
        else:
            # No updates, est = pred
            x_est[k] = x_pred[k]
            P_est[k] = P_pred[k]

            
        if k < N - 1:
            pred_timer[k] = time.time()
            x_pred[k+1], P_pred[k+1] = eskf.predict(
                                            x_est[k],
                                            P_est[k],
                                            # acc_t[k],
                                            z_acc[k], #Denne er k pga måten dataen er laget på (?)
                                            z_gyro[k+1],
                                            Ts_IMU[k],
                                            use_UDU
                                             )
            elapsed_pred_timer[k] = time.time() - pred_timer[k]
            
    result_UDU = (
            x_pred,
            x_est,
            P_est,
            GNSSk,
            elapsed_pred_timer,
            elapsed_est_timer       
              )


    return result_UDU