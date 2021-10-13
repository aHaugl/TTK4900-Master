# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:38:45 2021

@author: Andreas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:05:48 2021

@author: andhaugl
"""
# %% imports
from typing import Tuple, Sequence, Any
from dataclasses import dataclass, field
from cat_slice import CatSlice

import numpy as np
import scipy.linalg as la

# import sys

from quaternion import (
    euler_to_quaternion,
    quaternion_product,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)


# from state import NominalIndex, ErrorIndex
from utils import cross_product_matrix

from timer import*

# %% indices
POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)
ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)

@dataclass
class ESKF:
    sigma_acc: float #acc_std
    sigma_gyro: float #rate_std

    sigma_acc_bias: float   #cont_acc_bias_driving_noise_std
    sigma_gyro_bias: float  #cont_rate_bias_driving_noise_std

    p_acc: float = 0 #p_acc
    p_gyro: float = 0 #p_gyro

    S_a: np.ndarray = np.eye(3)
    S_g: np.ndarray = np.eye(3)
    debug: bool = True
    use_pseudorange: bool = True

    g: np.ndarray = np.array([0, 0,9.82])

    Q_err: np.array = field(init=False, repr=False)
    
    Q_err: np.array = field(init=False, repr=False)

    def __post_init__(self):
        if self.debug:
            print(
                "ESKF in debug mode, some numeric properties are checked at the expense of calculation speed"
            )
        

        self.Q_err = (
            la.block_diag(
                self.sigma_acc * np.eye(3), #rate_std
                self.sigma_gyro * np.eye(3), #gyro_std
                self.sigma_acc_bias * np.eye(3), #cont_acc_bias_driving_noise_std
                self.sigma_gyro_bias * np.eye(3), #cont_rate_bias_driving_noise_stdcont_rate_bias_driving_noise_std
            )
            ** 2
        )
            
    def predict_nominal(self,
                         x_nominal: np.ndarray,
                         acceleration_b: np.ndarray,
                         omega: np.ndarray,
                         Ts: float
                         ) -> np.ndarray:
        
        """Discrete time prediction,
           equation (10.58) in sensor fusion and/or Ch 5.3.2/eq 236 in Sola

        Args:
        -----------
            x_nominal (np.ndarray): The nominal state to predict, shape (16,)
            acceleration (np.ndarray): The estimated acceleration in body for the predicted interval, shape (3,)
            omega (np.ndarray): The estimated rotation rate in body for the prediction interval, shape (3,)
            Ts (float): The sampling time

        Raises:
        -----------
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
        -----------
            np.ndarray: The predicted nominal state, shape (16,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict_nominal: x_nominal incorrect shape {x_nominal.shape}"
        assert acceleration_b.shape == (
            3,
        ), f"ESKF.predict_nominal: acceleration_b incorrect shape {acceleration_b.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.predict_nominal: omega incorrect shape {omega.shape}"
        
        # Extract states
        position = x_nominal[POS_IDX]
        velocity = x_nominal[VEL_IDX]
        quaternion = x_nominal[ATT_IDX]
        acceleration_bias = x_nominal[ACC_BIAS_IDX]
        gyroscope_bias = x_nominal[GYRO_BIAS_IDX]
        
        if self.debug:
          assert np.allclose(
              np.linalg.norm(quaternion), 1, rtol=0, atol=1e-15
          ), "ESKF.predict_nominal: Quaternion not normalized."
          assert np.allclose(
              np.sum(quaternion ** 2), 1, rtol=0, atol=1e-15
          ), "ESKF.predict_nominal: Quaternion not normalized and norm failed to catch it."

        # Ts = 0.1
        R = quaternion_to_rotation_matrix(quaternion, debug=self.debug)
        g = np.array([0,0,9.82 ])
        
        # acceleration_world = R @ acceleration_b + g
        acceleration_world = acceleration_b + g #acceleration_b = specific_force, 
        # omega_world = R @ omega
        
        # print(position_prediction)
        velocity_prediction = velocity + Ts * acceleration_world
        # print("Ts: ", Ts)
        # print("velocity: ", velocity)
        # print("velocity_prediction: ", velocity_prediction)
        
        position_prediction = position + Ts * velocity + (Ts **2)/2 * acceleration_world
        # print("position: ", position)
        # print("position_prediction: ", position_prediction)
        omega_step = Ts * omega
        # print("omega_step:", omega_step)
        omega_step_norm = la.norm(omega_step)
        # print("omega_step_norm:", omega_step_norm)
        
        if omega_step_norm > 1e-15:
            # print("omega_step_norm > 0.01")
            delta_quat = np.array([np.cos(omega_step_norm/2 ),
                                *(np.sin(omega_step_norm/2 )* omega_step.T / omega_step_norm)])
        
        else:
            # print("omega_step_norm >= 0.01")
            delta_quat = np.array([np.cos(omega_step_norm/2 ),
                                *(np.sin(omega_step_norm/2 )* omega_step.T / 1)])

        # print("quaternion: ", quaternion)
        # print("delta_quat: ", delta_quat)
        quaternion_prediction = quaternion_product(quaternion, delta_quat)
        # print("quat_pred:", quaternion_prediction)
        
        #Quaternion normalization
        quaternion_prediction = (quaternion_prediction
                                 / la.norm(quaternion_prediction))
        
        #1. Ordens approx
        # Cont eq: acc_bias_dot = -p_acc*I*acc_bias
        # Disc eq: acc_bias_k+1 = acc_bias_k - p_acc*Ts*acc_bias_true, => acc_bt = acc_est at best
        acceleration_bias_prediction = ((1 - Ts * self.p_acc)
                                        * acceleration_bias) 
        
        gyroscope_bias_prediction = ((1 - Ts * self.p_gyro)
                                     * gyroscope_bias)
        
        
        x_nominal_predicted = np.concatenate(
            (
                position_prediction,
                velocity_prediction,
                quaternion_prediction,
                acceleration_bias_prediction,
                gyroscope_bias_prediction,
            )
        )
        
        assert x_nominal_predicted.shape == (
            16,
        ), f"ESKF.predict_nominal: x_nominal_predicted shape incorrect {x_nominal_predicted.shape}"
        return x_nominal_predicted
    
    def Aerr(
            self,
            x_nominal: np.ndarray,
            acceleration: np.ndarray,
            omega: np.ndarray,
            ) -> np.ndarray:
        """Calculates the continous time error state dynamics jacobian

        Parameters
        ----------
        x_nominal : np.ndarray
            Nominal state vector.
        acceleration : np.ndarray
            Estimated acceleration in body for prediction interval, (3,).
        omega : np.ndarray
            Estimated rotation rate in body for prediction interval, (3,).
            
        Raises
        -------
        AssertionError: If any input or output is wrong shape.
        
        Returns
        -------
        Aerr: Continous time error state dynamics Jacobian (15,15), correspon

        """
        assert x_nominal.shape == (
            16,
        ), f"ESKF.Aerr: x_nominal incorrect shape {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.Aerr: acceleration incorrect shape {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.Aerr: omega incorrect shape {omega.shape}"
        
        #Rotation matrix
        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug = self.debug)
        
        #Allocate matrix
        A = np.zeros((15,15))
        
        #Set submatrices
        A[POS_IDX * VEL_IDX] = np.eye(3)
        A[VEL_IDX * ERR_ATT_IDX] = -R @ cross_product_matrix(acceleration)
        A[VEL_IDX * ERR_ACC_BIAS_IDX] = -cross_product_matrix(omega)
        A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] = -np.eye(3)
        A[ERR_ACC_BIAS_IDX * ERR_ACC_BIAS_IDX] = -self.p_acc * np.eye(3)
        A[ERR_GYRO_BIAS_IDX * ERR_GYRO_BIAS_IDX] = -self.p_gyro * np.eye(3)
        
        #Bias Correction
        A[VEL_IDX * ERR_ACC_BIAS_IDX] = A[VEL_IDX * ERR_ACC_BIAS_IDX] @ self.S_a
        A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] = (
            A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] @ self.S_g
        )
    
        assert A.shape ==(
            15,
            15,
            ), f"ESKF.Aerr: A-error matrix shape incorrect {A.shape}"
        return A
    
    def Gerr(self,
             x_nominal: np.ndarray,
             ) -> np.ndarray:
        """Calculate the continous time error state noise input matrix
    

        Parameters
        ----------
        x_nominal : np.ndarray
            Nominal state vector (16,)
         : TYPE
            np.ndarray.

        Returns
        -------
        G : TYPE
            DESCRIPTION.

        """
        assert x_nominal.shape == (
            16,
        ), f"ESKF.Gerr: x_nominal incorrect shape {x_nominal.shape}"
        
        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)
        
        G = np.zeros((15,12))
        G[3:] = la.block_diag(-R, np.eye(3), np.eye(3), np.eye(3))
        
        assert G.shape == (
            15,
            12,
            ), f"ESKF.Gerr: G-matrix shape incorrect {G.shape}"
        return G
    
    def discrete_error_matrices(
            self,
            x_nominal: np.ndarray,
            acceleration: np.ndarray,
            omega: np.ndarray,
            Ts: float,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the discrete time linearized error state transition and covariance matrix
        

        Parameters
        ----------
        x_nominal : np.ndarray
            Nominal state vector.
        acceleration : np.ndarray
            Estimated acceleration in body for prediction interval, (3,).
        omega : np.ndarray
            Estimated rotation rate in body for prediction interval, (3,).
        Ts : float
            The ampling time.
        
        Raises
        -------
        AssertionError: If any input or output is wrong shape.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]: Discrete error matrices (Tuple, Ad, GQGd)
            Ad: Discrete time error state system matrix (15,15)
            GQGd: Discrete time noise covariance matrix (15,15)

        """
        assert x_nominal.shape == (
            16,
        ), f"ESKF.discrete_error_matrices: x_nominal incorrect shape {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.discrete_error_matrices: acceleration incorrect shape {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.discrete_error_matrices: omega incorrect shape {omega.shape}"
        
        
        #Calculate continious time error state dynamics Jacobian
        A = self.Aerr(x_nominal, acceleration, omega)
        # print("Aerr_A: ", A)
        #Calculate continuous time error state noise input matrix
        G = self.Gerr(x_nominal)
        # print("Gerr_G: ", G)
        
        V = np.block([[-A, G @ self.Q_err @ G.T],
                      [np.zeros_like(A), A.T]]) * Ts
        
        assert V.shape == (
            30,
            30,
            ), f"ESKF.discrete_error_matrices: Van Loan matrix shape incorrect {V.shape}"
        VanLoanMatrix = la.expm(V)
        # VanLoanMatrix = np,identity(V.shape[0]) + V #Fast but unsafe
        
        Ad = VanLoanMatrix[CatSlice(15, 30)**2].T
        GQGd = Ad @ VanLoanMatrix[CatSlice(0, 15) * CatSlice(15, 30)]
        
        assert Ad.shape == (
            15,
            15,
            ), f"ESKF.discrete_error_matrices: Ad-matrix shape incorrect {Ad.shape}"
        assert GQGd.shape == (
            15,
            15,
            ), f"ESKF.discrete_error_matrices: GQGd-matrix shape incorrect {GQGd.shape}"
        
        return Ad, GQGd
    
    def predict_covariance(
            self,
            x_nominal: np.ndarray,
            P: np.ndarray,
            acceleration: np.ndarray,
            omega: np.ndarray,
            Ts: float,
            ) -> np.ndarray:
        
        """Predicts the error state covariance Ts time units ahead using linearized
        continous time dynamics
        
        Args:
            x_nominal: nominal state (16,)
            P: Error state covariance (15,15)
            acceleration: Estimated acceleration for prediction interval (3,)
            omega: Estimated rotation rate for prediction interval (3,)
            Ts: Sampling time
        Raises:
            AssertionError: If inputs or output is wrong shape.
        Returns:
            The predicted error state covariance matrix (15,15)
        """
        assert x_nominal.shape ==(
            16,
            ), f"ESKF.predict_covariance: x_nominal shape incorrext {x_nominal.shape}"
        assert P.shape ==(
            15,
            15,
            ), f"ESKF.predict_covariance: P shape incorrect {P.shape}"
        assert acceleration.shape ==(
            3,
            ), f"ESKF.predict_covariance: acceleration shape inncorrect {acceleration.shape}"
        assert omega.shape ==(
            3,
            ), f"ESKF.predict_covariance: omega shape inncorrect {omega.shape}"
        

        
        #Compute discrete time linearized error state transition and covariance matrix
        Ad, GQGd = self.discrete_error_matrices(
            x_nominal,
            acceleration,
            omega,
            Ts)
        # print("GQGd from predict_covariance: ", GQGd)
        
        P_predicted = Ad @ P @ Ad.T + GQGd
        
        # print(P_predicted[ERR_ATT_IDX,ERR_ATT_IDX])
        
        assert P_predicted.shape == (
            15,
            15,
        ), f"ESKF.predict_covariance: P_predicted shape incorrect {P_predicted.shape}"
        
        return P_predicted
    
    
    def predict(self,
                x_nominal: np.ndarray,
                P: np.ndarray,
                z_acc: np.ndarray,
                z_gyro: np.ndarray,
                Ts: float,
                ) -> np.array:#Tuple [np.array, np.array]:
        """
        
        Parameters
        ----------
        x_nominal : np.ndarray
            Nominal state to predict, (16,).
        P : np.ndarray
            Error state covariance to predict (15,15).
        z_acc : np.ndarray
            Measured acceleration for prediction interval, (3,).
        z_gyro : np.ndarray
            Measured rotation rate for the prediction interval, (3,).
        Ts : float
            The sampling time.
        Raises
        -------
        AssertionError: If any input or output is wrong shape
        
        Returns
        -------
        TYPE
            Tuple[np.array, np.array]: Prediction Tuple(x_nominal_predicted,
                                                        P_predicted)
        x_nominal predicted:
                The predicted nominal state, (16,)
        P_predicted :
            The predicted error state covariance (15,15).

        """
        assert x_nominal.shape == (
            16,
            ), f"ESKF.predict: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (
            15,
            15,
            ), f"ESKF.predict: P matrix shape incorrect {P.shape}"
        assert z_acc.shape == (
            3,
            ), f"ESKF.predict: z_acc shape incorrect {z_acc.shape}"
        assert z_gyro.shape == (
            3,
            ), f"ESKF.predict: z_gyro shape incorrect {z_gyro.shape}"

        #Correct measurement. In this case S_a = S_g = eye(3)
        r_z_acc = self.S_a @ z_acc
        r_z_gyro = self.S_g @ z_gyro
        
        #Debiased IMU measurements
        acc_bias = x_nominal[ACC_BIAS_IDX]
        gyro_bias = x_nominal[GYRO_BIAS_IDX]
        
        acceleration = r_z_acc - acc_bias
        omega = r_z_gyro - gyro_bias
        
        #Predict:
        # print("ESKF.predict quaternion: ", x_nominal[ATT_IDX])
        x_nominal_predicted = self.predict_nominal(x_nominal,
                                                   acceleration,
                        
                                                   omega,
                                                   Ts
                                                   )
        
        P_predicted = self.predict_covariance(x_nominal,
                                              P,
                                              acceleration,
                                              omega,
                                              Ts)
        # print("x_nominal_predicted[k+1]: ", x_nominal_predicted[0:6])
        assert x_nominal_predicted.shape ==(
            16,
        ), f"ESKF.predict: x_nominal_predicted shape incorrect {x_nominal_predicted.shape}"
        assert P_predicted.shape == (
            15,
            15
        ), f"ESKF.predict: P_predicted_nominal shape incorrect {P_predicted.shape}"
        
        return x_nominal_predicted, P_predicted
    
    def inject(self,
               x_nominal: np.ndarray,
               delta_x: np.ndarray,
               P: np.ndarray,
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        
        Parameters
        ----------
        x_nominal : np.ndarray
            Nominal state to inject the error state deviation into, (16,).
        delta_x: np.ndarray 
            Error state deviation shape (15,)
        P : np.ndarray
            Error state covariance matrix (15,15).
        z_acc : np.ndarray
            Measured acceleration for prediction interval, (3,).
        z_gyro : np.ndarray
            Measured rotation rate for the prediction interval, (3,).
        Ts : float
            The sampling time.
            
        Raises
        -------
        AssertionError: If any input or output is wrong shape

        Returns
        -------
        x_injected : TYPE
            The injected nominal state, (16,).
        P_injected : TYPE
            The injected error state covariance matrix, (15,15)

        """
        assert x_nominal.shape == (
            16,
            ), f"ESKF.inject: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape ==(
            15,
            15,
            ), f"ESKF.inject: P shape incorrect {P.shape}"
        if delta_x.shape != (15,):
            delta_x = np.reshape(delta_x, ((15,)))
        
        x_injected = x_nominal.copy()
        
        #Inject error state into nominal state (except attitude)
        # print(x_injected.shape)
        # print(delta_x.shape)
        
        x_injected[POS_IDX +
                   VEL_IDX + 
                   ACC_BIAS_IDX + 
                   GYRO_BIAS_IDX] = (x_nominal[POS_IDX +
                                              VEL_IDX + 
                                              ACC_BIAS_IDX + 
                                              GYRO_BIAS_IDX]
                                               +
                                   delta_x[POS_IDX +
                                           VEL_IDX +
                                           ERR_ACC_BIAS_IDX +
                                           ERR_GYRO_BIAS_IDX]
                                    )
                                               
        #Inject attitude                                               
        delta_quat = np.array([1, *delta_x[ERR_ACC_BIAS_IDX]/2])
        x_injected[ATT_IDX] = quaternion_product(x_nominal[ATT_IDX],
                                                 delta_quat)
        
        #Normalize quaternion
        x_injected[ATT_IDX] = x_injected[ATT_IDX] / la.norm(x_injected[ATT_IDX])
    
        #Covariance reset eq 3.20
        # Compensate for injection in the covariances
        
        G_injected = (
            la.block_diag(np.eye(6),
                          np.eye(3) - cross_product_matrix(delta_x[ERR_ATT_IDX]/2),
                          np.eye(6)))
        P_injected = G_injected @ P @ G_injected.T
        
        assert x_injected.shape ==(
            16,
            ), f"ESKF.inject: x_injected shape incorrect {x_injected.shape}"
        assert P_injected.shape ==(
            15,
            15,
            ), f"ESKF.inject: P_injected shape incorret {P_injected.shape}"
        
        return x_injected, P_injected
    
    

    def update_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        R_beacons: np.ndarray,
        beacon_location: np.ndarray,
        use_batch_pseudoranges,
        use_iterative_pseudoranges,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates the state and covariance from a GNSS position measurement

        Parameters:
        -------
            x_nominal (np.ndarray): The nominal state to update, shape (16,)
            P (np.ndarray): The error state covariance to update, shape (15, 15)
            z_GNSS_position (np.ndarray): The measured 3D position, shape (3,)
            R_GNSS (np.ndarray): The estimated covariance matrix of the measurement, shape (3, 3)
            lever_arm (np.ndarray, optional): The position of the GNSS receiver from the IMU reference, shape (3,). Defaults to np.zeros(3), shape (3,).

        Raises:
        -------
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
        -------
            Tuple[np.ndarray, np.ndarray]: Updated Tuple(x_injected, P_injected):
                x_injected: The nominal state after injection of updated error state, shape (16,)
                P_injected: The error state covariance after error state update and injection, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.update_GNSS: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (
            15, 15), f"ESKF.update_GNSS: P shape incorrect {P.shape}"
        assert z_GNSS_position.shape == (
            3,
        ), f"ESKF.update_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
        assert R_GNSS.shape == (
            3,
            3,
        ), f"ESKF.update_GNSS: R_GNSS shape incorrect {R_GNSS.shape}"
        assert lever_arm.shape == (
            3,
        ), f"ESKF.update_GNSS: lever_arm shape incorrect {lever_arm.shape}"

        
        if (use_batch_pseudoranges):
            
            delta_x, P_update = self.batch_pseudorange(                                                x_nominal,
                                                    z_GNSS_position,
                                                    P,
                                                    R_GNSS,
                                                    beacon_location,
                                                    R_beacons
                                                    ) 
        elif(use_iterative_pseudoranges):
            delta_x, P_update = self.iterative_pseudorange_v2(                                                x_nominal,
                                            z_GNSS_position,
                                            P,
                                            R_GNSS,
                                            beacon_location,
                                            R_beacons
                                            ) 
        
        x_injected, P_injected = self.inject(x_nominal, delta_x, P_update)
        
        

        
        # delta_x_i, P_update_i = self.iterative_pseudorange(x_nominal,
        #                                         z_GNSS_position,
        #                                         P,
        #                                         R_GNSS,
        #                                         beacon_location,
        #                                         R_beacons
        #                                         ) 
        # print(delta_x - delta_x_i)
        # print("delta_x iterative : ", delta_x_i, delta_x_i.shape)
        # delta_x_i = delta_x_i.reshape((-1,))
        # print("delta_x iterative : ", delta_x_i, delta_x_i.shape)
        
        # print("delta_x bactch: ", delta_x, delta_x.shape)
        # print("delta_x iterative : ", delta_x_i, delta_x_i.shape)
        # print("P_update bactch: ", P_update, P_update.shape)
        # print("P_update iterative: ", P_update_i, P_update_i.shape)
        
        
        

        # error state injection

        # x_injected, P_injected = self.inject(x_nominal, delta_x_i, P_update_i)
        # print("x_injected:", x_injected[POS_IDX])

        assert x_injected.shape == (
            16,
        ), f"ESKF.update_GNSS: x_injected shape incorrect {x_injected.shape}"
        assert P_injected.shape == (
            15,
            15,
        ), f"ESKF.update_GNSS: P_injected shape incorrect {P_injected.shape}"

        return x_injected, P_injected
  
    def batch_pseudorange(self,
                      x_nominal: np.ndarray,
                      # x_true: np.ndarray,
                      z_GNSS_position: np.ndarray,
                      P:np.ndarray,
                      # S:np.ndarray, 
                      R_GNSS:np.ndarray,
                      b_loc: np.ndarray,
                      R_beacons: np.ndarray) -> np.ndarray:
        
        """
        Generate pseudoranges and design matrix H whenever a GNSS measurement
        is recieved (1Hz).
        
        Is used in the update step where the filter incorporates the predicted values and 
        the information from the measurements to improve the estimated position errors.
        
        This is done in the functions Update_GNSS_position
        """
        #tic()
        
        I = np.eye(*P.shape)
        
        num_beacons = len(b_loc)
        # num_beacons = 3
        est_ranges = np.zeros(num_beacons)
        measured_ranges = np.zeros(num_beacons)
        delta_P = np.zeros(num_beacons)
        
        pos_est = x_nominal[POS_IDX]  
        pos_meas = z_GNSS_position
        
        #ranges/LOS vectors
        for i in range(num_beacons):
            est_ranges[i] = np.array(
                [la.norm(-pos_est + b_loc[i])]
                )
            measured_ranges[i] = np.array(
                [la.norm(-pos_meas + b_loc[i])]
                )
                        
        #Pseudorange measurement residual
        v = measured_ranges - est_ranges
        # print("delta_P: ", delta_P)
        
        #Geometry matrix consisting of normalized LOS-vectors
        H = np.zeros((num_beacons, 15))
        for k in range(num_beacons):
              for j in range(3):
                  # print(j)
                  # print("Range for this beacon: ", ranges[k])
                 
                  H[k,j] = (b_loc[k,j] - pos_est[j])#/ranges[k]
              # print("H[k]: ", H[k])
              H[k] = -(H[k])/est_ranges[k]

        # print(H)
        S = H @ P @ H.T + R_beacons[:num_beacons, :num_beacons] #R_GNSS
        # print(S.shape)

        
        W = P @ H.T @ la.inv(S)
        # print(W,W.shape)

        delta_x = W @ v
        
        Jo = I - W @ H  # for Joseph form
        
        # Update the error covariance

        P_update = Jo @ P @ Jo.T + W @ R_beacons[:num_beacons, :num_beacons] @ W.T

        return delta_x, P_update
    
    
    def iterative_pseudorange_v2(self,
                      x_nominal: np.ndarray,
                      # x_true: np.ndarray,
                      z_GNSS_position: np.ndarray,
                      P:np.ndarray,
                      # S:np.ndarray, 
                      R_GNSS:np.ndarray,
                      b_loc: np.ndarray,
                      R_beacons: np.ndarray) -> np.ndarray:
        
        """
        Generate pseudoranges and design matrix H whenever a GNSS measurement
        is recieved (1Hz).
        
        Is used in the update step where the filter incorporates the predicted values and 
        the information from the measurements to improve the estimated position errors.
        
        This is done in the functions Update_GNSS_position
        """
        #tic()
        
        I = np.eye(*P.shape)
        
        num_beacons = len(b_loc)
        # num_beacons = 1
        num_beacons_at_the_time = 1
        est_ranges = np.zeros(num_beacons)
        measured_ranges = np.zeros(num_beacons)
        delta_P = np.zeros(num_beacons)
        
        pos_est = x_nominal[POS_IDX]  
        pos_meas = z_GNSS_position
        
        delta_x = np.zeros((15,))
        #ranges/LOS vectors
        for i in range(num_beacons):
            est_ranges[i] = np.array(
                [la.norm(-pos_est + b_loc[i])]
                )
            measured_ranges[i] = np.array(
                [la.norm(-pos_meas + b_loc[i])]
                )
                        
        #Pseudorange measurement residual
        v = measured_ranges - est_ranges
        # print("delta_P: ", delta_P)
        
        #Geometry matrix consisting of normalized LOS-vectors
        for i in range(num_beacons):
            # print(i)
            H = np.zeros((num_beacons_at_the_time, 15))
            for k in range(num_beacons_at_the_time):
                  for j in range(3):
                      # print(j)
                      # print("Range for this beacon: ", ranges[k])
                     
                      H[k,j] = (b_loc[k+i,j] - pos_est[j])#/ranges[k]
                  # print("H[k]: ", H[k])
                  H[k] = -(H[k])/est_ranges[k+i]
    
            # print(H)
            # print(P)
            # print(R_beacons)
            #Innovation covariance: 1x1
            S = H @ P @ H.T + R_beacons[i, i] #R_GNSS
            # print(S)
            
            #Kalman gain 15x1
            W = P @ H.T/S # @ la.inv(S)
            
    
            delta_x = (W * v[i]).T #Adding up the error states
            # print(delta_x, delta_x.shape)
            
            # for Joseph form
            Jo = I - W @ H  
            # Update the error covariance
            P = Jo @ P @ Jo.T + W @  W.T * R_beacons[i, i]
            
            
    
        P_update = P
    
        return delta_x, P_update
    
         
    # def iterative_pseudorange(self,
    #                   x_nominal: np.ndarray,
    #                   # x_true: np.ndarray,
    #                   z_GNSS_position: np.ndarray,
    #                   P:np.ndarray,
    #                   # S:np.ndarray, 
    #                   R_GNSS:np.ndarray,
    #                   b_loc: np.ndarray,
    #                   R_beacons: np.ndarray) -> np.ndarray:
        
    #     """This is made to work with 3 beacons at the time
    #     Generate pseudoranges and design matrix H whenever a GNSS measurement
    #     is recieved (1Hz).
        
    #     x_nominal: Current state prediction, size((15,1))
    #     P: Predicted covariance at this time step, size((15,15))
        
    #     """
    #     #tic()
    #     assert x_nominal.shape == (
    #         16,
    #     ), f"ESKF.update_GNSS: x_nominal shape incorrect {x_nominal.shape}"
    #     assert P.shape == (
    #         15, 15), f"ESKF.update_GNSS: P shape incorrect {P.shape}"
    #     assert z_GNSS_position.shape == (
    #         3,
    #     ), f"ESKF.update_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
        
    #     I = np.eye(*P.shape)
        
    #     # num_beacons = len(b_loc)
    #     num_beacons = 3
    #     est_ranges = np.zeros(num_beacons)
    #     measured_ranges = np.zeros(num_beacons)
    #     delta_P = np.zeros(num_beacons)
        
    #     pos_est = x_nominal[POS_IDX]  
    #     pos_meas = z_GNSS_position

    #     # %% Iterativ 3 og 3 beacons         #ranges/LOS vectors
    #     for i in range(num_beacons):
    #         est_ranges[i] = np.array(
    #             [la.norm(-pos_est + b_loc[i])]
    #             )
    #         measured_ranges[i] = np.array(
    #             [la.norm(-pos_meas + b_loc[i])]
    #             )
                        
    #     #Pseudorange measurement residual
    #     v = measured_ranges - est_ranges
    #     # Geometry matrix consisting of normalized LOS-vectors for each beacon?
    #     delta_x = np.zeros((15,))
    #     H_all = np.zeros((num_beacons,15))
    #     W_all = np.zeros((15,num_beacons))
    #     H = np.zeros((3,15))
    #     b_sel = 0 #To indicate which beacons are in use
        
    #     for i in range(int(num_beacons / 3)):#Divide loop into parts of 3: Not handling numbeacons not divisible by 3
    #         # print("i: ", i)
    #         for k in range(b_sel, b_sel + 3): #Iterating over 3 and 3 beacons
    #             # print("k: ",k)
    #             # print("Currently using beacon number: ", k)
    #             # for i in range(3): 
    #             for j in range(3):
    #                 H[k-b_sel,j] = (b_loc[k-b_sel,j] - pos_est[j])
    #                 # print((b_loc[k,j] - pos_est[j]))
                
    #             H[k-b_sel] = -H[k-b_sel]/est_ranges[k]
    #             H_all[b_sel:b_sel+3,:] = H
            
    #         # print("H[k]: ", H, H.shape)
    #         # print("Accumulated H: ", H_all, H.shape)
            
    #         S = H @ P @ H.T + R_GNSS # R_beacons[a:a+3, a:a+3]
    #         W = P @ H.T @ la.inv(S)
            
    #         # W_all[:,b_sel:b_sel+3] = W
    #         # print("Accumulated gain matrix: ", W_all, W_all.shape)
            
    #         # print("S: ", S, S.shape)
    #         # print("W: ", W, W.shape)
    #         # print("a: ", a)
    #         print("v[a:a+3]: ", v[b_sel:b_sel+3], v[b_sel:b_sel+3].shape)
            
    #         delta_x =  W @ v[b_sel:b_sel+3]
    #         # print("Accumulated delta_x: " ,delta_x, delta_x.shape)
            
    #         #Må gjøre en korreksjon av covariansen for den neste batchen blir gjennomført
    #         #Dette blir ustabilt
    #         # Jo = I - W @ H
    #         # P = Jo @ P @ Jo.T + W @ R_beacons[a:a+3, a:a+3] @ W.T
            
    #         if not ((num_beacons - b_sel) % 3):
    #             print("More beacons available, selecting three next")
    #             b_sel += 3
    #         elif ((b_sel + 3) > num_beacons):
    #             print("There are less than 3 beacons available. Using the final 3 beacons")
    #             b_sel = num_beacons - 3 
    #         print("b_sel: ", b_sel)
            
    #     # P_update = P
    #     #Dette funker, men er ekvivalent med å kun gjøre korreksjon med 3 beacons
    #     Jo = I - W @ H  # for Joseph form
        
    #     # Update the error covariance

    #     P_update = Jo @ P @ Jo.T + W @ R_GNSS @ W.T #R_beacons[:3,:3]

    #     # Jo = I - W_all @ H_all
    #     # P_update = Jo @ P @ Jo.T + W_all @ R_beacons[:num_beacons, :num_beacons] @ W_all.T
                
    #     return delta_x, P_update

    # def iterative_pseudorange(self,
    #                   x_nominal: np.ndarray,
    #                   # x_true: np.ndarray,
    #                   z_GNSS_position: np.ndarray,
    #                   P:np.ndarray,
    #                   # S:np.ndarray, 
    #                   R_GNSS:np.ndarray,
    #                   b_loc: np.ndarray,
    #                   R_beacons: np.ndarray) -> np.ndarray:
        
    #     """
    #     Generate pseudoranges and design matrix H whenever a GNSS measurement
    #     is recieved (1Hz).
        
    #     Is used in the update step where the filter incorporates the predicted values and 
    #     the information from the measurements to improve the estimated position errors.
        
    #     This is done in the functions Update_GNSS_position
    #     """
    #     #tic()
        
    #     I = np.eye(*P.shape)
        
    #     num_beacons = len(b_loc)
    #     est_ranges = np.zeros(num_beacons)
    #     measured_ranges = np.zeros(num_beacons)
    #     delta_P = np.zeros(num_beacons)
        
    #     pos_est = x_nominal[POS_IDX]  
    #     pos_meas = z_GNSS_position


    #     # %% Iterativ 
    #     #ranges/LOS vectors
    #     for i in range(num_beacons):
    #         est_ranges[i] = np.array(
    #             [la.norm(-pos_est + b_loc[i])]
    #             )
    #         measured_ranges[i] = np.array(
    #             [la.norm(-pos_meas + b_loc[i])]
    #             )
                        
    #     #Pseudorange measurement residual
    #     # v = np.array((15,1))
    #     v = measured_ranges - est_ranges
    #     # v = np.reshape(v, ((15,1)))
    #     # print("delta_P: ", delta_P)
        
    #     # Geometry matrix consisting of normalized LOS-vectors for each beacon?

    #     P_update = np.zeros((15,15))
    #     Jo = np.zeros((15,15))
    #     R = np.zeros((15,15))
    #     delta_x = np.zeros((15,1))
    #     W1 = np.zeros((15,15))
    #     H1 = np.zeros((15,15))
    #     for k in range(num_beacons):
    #         # print("k: ",k)
    #         H = np.zeros((1,15))
    #         for j in range(3):
    #               H[0,j] = (b_loc[k,j] - pos_est[j])#/ranges[k]
    #         H = -H/est_ranges[k]
    #         H1[k,:] = H
    #         print("H1: ", H1, H1.shape)
            
    #         # print(H[k] @ P @ H[k].T)
    #         #S = scalar            
            
    #         # S = np.dot(H,P,H.T)
    #         # print(P)
    #         S = H @ P @ H.T #+ R_beacons[k,k] #Dette blir ein skalar. Skal det være ein vektor? 
    #         # print("P",P)
    #         # W = P @ H.T / S# + R_beacons[k,k]
    #         W = P* H / S# + R_beacons[k,k]
    #         # W1[:,k] = W.T
    #         # print("W",W)
            
    #         print(delta_x)
    #         delta_x = delta_x + (W * v[k])
    #         print(delta_x,delta_x.shape)
            
    #     I = np.eye(*P.shape)
    #     Jo = I - W1 @ H1  # for Joseph form
    #     P = Jo @ P @ Jo.T + W1 @ R_beacons @ W1.T
    #     P_update = P_update + P
            
    #     delta_x = np.reshape(delta_x,((15,)))
        
        
    #     return delta_x, P_update
        # return x_injected, P_injected

             
              # I = np.eye(*P.shape)
              # Jo = I - W @ H  # for Joseph form
        
              # # Update the error covariance
              # P = Jo @ P @ Jo.T + W @ R_GNSS @ W.            
        # P_update = P
        # Gjøre korreksjon fra linje 570
        # x_est = x_est + delta_x

    
    # @classmethod 
    # def delta_x(cls,
    #             x_nominal: np.ndarray,
    #             x_true: np.ndarray
    #             ) -> np.ndarray:
    #     """ Calculates the error state between x_nominal and x_true
        

    #     Parameters
    #     ----------
    #     x_nominal : np.ndarray
    #         Nominal estimated state, (16,)
    #     x_true : np.ndarray
    #         The true state vector, (16,)
            
    #     Raises
    #     -------
    #     AssertionError: If any input or output is wrong shape.
        
    #     Returns
    #     -------
    #     d_x: 
    #         Error state state between x_nominal and x_true

    #     """
    #     assert x_nominal.shape == (
    #         16,
    #         ), f"ESKF.delta_x: x_nominal shape incorrect {x_nominal.shape}"
    #     assert x_true.shape == (
    #         16,
    #         ), f"ESKF.delta_x: x_nominal shape incorrect {x_true.shape}"
        
       
    #     delta_position = x_true[POS_IDX] - x_nominal[POS_IDX]
            
    #     delta_velocity = x_true[VEL_IDX] - x_nominal[VEL_IDX]
        
    #     # Conjugate of quaternion
    #     quat_conj = x_nominal[ATT_IDX]
    #     quat_conj[1:] *= -1
        
    #     delta_quat = quaternion_product(quat_conj,
    #                                     x_true[ATT_IDX])
    #     delta_theta = 2* delta_quat[1:]
        
    #     delta_bias = (
    #         x_true[ACC_BIAS_IDX + GYRO_BIAS_IDX] -  x_nominal[ACC_BIAS_IDX + GYRO_BIAS_IDX]
    #         )
        
        
    #     d_x = np.concatenate(
    #         (
    #         delta_position,
    #         delta_velocity,
    #         delta_theta,
    #         delta_bias
    #         )
    #     )
    #     # print("d_x is estimated as: ", d_x)

    #     assert d_x.shape == (
    #         15,), f"ESKF.delta_x: d_x shape incorrect {d_x.shape}"

    #     return d_x

# %% Backup of working code
    # def iterative_pseudorange_v2(self,
    #                   x_nominal: np.ndarray,
    #                   # x_true: np.ndarray,
    #                   z_GNSS_position: np.ndarray,
    #                   P:np.ndarray,
    #                   # S:np.ndarray, 
    #                   R_GNSS:np.ndarray,
    #                   b_loc: np.ndarray,
    #                   R_beacons: np.ndarray) -> np.ndarray:
        
    #     """
    #     Generate pseudoranges and design matrix H whenever a GNSS measurement
    #     is recieved (1Hz).
        
    #     Is used in the update step where the filter incorporates the predicted values and 
    #     the information from the measurements to improve the estimated position errors.
        
    #     This is done in the functions Update_GNSS_position
    #     """
    #     #tic()
        
    #     I = np.eye(*P.shape)
        
    #     num_beacons = len(b_loc)
    #     # num_beacons = 1
    #     num_beacons_at_the_time = 1
    #     est_ranges = np.zeros(num_beacons)
    #     measured_ranges = np.zeros(num_beacons)
    #     delta_P = np.zeros(num_beacons)
        
    #     pos_est = x_nominal[POS_IDX]  
    #     pos_meas = z_GNSS_position
        
    #     #ranges/LOS vectors
    #     for i in range(num_beacons):
    #         est_ranges[i] = np.array(
    #             [la.norm(-pos_est + b_loc[i])]
    #             )
    #         measured_ranges[i] = np.array(
    #             [la.norm(-pos_meas + b_loc[i])]
    #             )
                        
    #     #Pseudorange measurement residual
    #     v = measured_ranges - est_ranges
    #     # print("delta_P: ", delta_P)
        
    #     #Geometry matrix consisting of normalized LOS-vectors
    #     for i in range(num_beacons):
    #         print(i)
    #         H = np.zeros((num_beacons_at_the_time, 15))
    #         for k in range(num_beacons_at_the_time):
    #               for j in range(3):
    #                   # print(j)
    #                   # print("Range for this beacon: ", ranges[k])
                     
    #                   H[k,j] = (b_loc[k+i,j] - pos_est[j])#/ranges[k]
    #               # print("H[k]: ", H[k])
    #               H[k] = -(H[k])/est_ranges[k+i]
    
    #         print(H)
    #         # print(P)
    #         # print(R_beacons)
    #         S = H @ P @ H.T + R_beacons[i, i] #R_GNSS
    #         print(S)
    
    #         W = P @ H.T @ la.inv(S)
    
    #         delta_x = W * v[i]
    #         print(delta_x, delta_x.shape)
            
    #         Jo = I - W @ H  # for Joseph form
            
    #         # Update the error covariance
    
    #         P_update = Jo @ P @ Jo.T + W @  W.T * R_beacons[i, i]
    
    #     return delta_x, P_update
    
    # def get_measurement_jacobian(self,
    #                              x_nominal: np.ndarray
    #                              ):
        
    #     """ Returns the (3, 15) shape matrix H = Hx @ X_delta_x """

    #     nominal_quaternion = x_nominal[ATT_IDX]
    #     eta = nominal_quaternion[0]
    #     eps1 = nominal_quaternion[1]
    #     eps2 = nominal_quaternion[2]
    #     eps3 = nominal_quaternion[3]

    #     # Create Q matrix (eq. 10.78)
    #     Q_delta_omega = np.array([
    #         [-eps1,     -eps2,      -eps3],
    #         [eta,       -eps3,      eps2],
    #         [eps3,      eta,        -eps1],
    #         [-eps2,     eps2,       eta]
    #     ]) / 2

    #     # Create X matrix (eq. 10.77)
    #     X_delta_x = np.zeros((16, 15))
    #     X_delta_x[(POS_IDX + VEL_IDX) ** 2] = np.eye(6)
    #     X_delta_x[ATT_IDX * ERR_ATT_IDX] = Q_delta_omega
    #     X_delta_x[(ACC_BIAS_IDX + GYRO_BIAS_IDX) * (ERR_ACC_BIAS_IDX + ERR_GYRO_BIAS_IDX)] = np.eye(6)

    #     # Create Hx matrix (eq. 10.80)
    #     Hx = np.eye(3, M=16)

    #     # Create H matrix (eq. 10.76)
    #     H = Hx @ X_delta_x

    #     # print("X_delta_x: " , X_delta_x)
    #     # print("Hx: " , Hx)
    #     # print("H: ", H)
        
    #     assert H.shape == (3, 15), 'H measurement jacobian not of shape (3,15)'
    #     return H
    
    # def innovation_GNSS_position(
    #     self,
    #     x_nominal: np.ndarray,
    #     P: np.ndarray,
    #     z_GNSS_position: np.ndarray,
    #     R_GNSS: np.ndarray,
    #     lever_arm: np.ndarray = np.zeros(3),
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Calculates the innovation and its covariance for a GNSS position measurement

    #     Parameters:
    #     -------
    #         x_nominal (np.ndarray): The nominal state to calculate the innovation from, shape (16,)
    #         P (np.ndarray): The error state covariance to calculate the innovation covariance from, shape (15, 15)
    #         z_GNSS_position (np.ndarray): The measured 3D position, shape (3,)
    #         R_GNSS (np.ndarray): The estimated covariance matrix of the measurement, shape (3, 3)
    #         lever_arm (np.ndarray, optional): The position of the GNSS receiver from the IMU reference. Defaults to np.zeros(3).

    #     Raises:
    #     -------
    #         AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

    #     Returns:
    #     -------
    #         Tuple[ np.ndarray, np.ndarray ]: Innovation Tuple(v, S):
    #             v: innovation, shape (3,)
    #             S: innovation covariance, shape (3, 3)
    #     """

    #     assert x_nominal.shape == (
    #         16,
    #     ), f"ESKF.innovation_GNSS: x_nominal shape incorrect {x_nominal.shape}"
    #     assert P.shape == (
    #         15, 15), f"ESKF.innovation_GNSS: P shape incorrect {P.shape}"

    #     assert z_GNSS_position.shape == (
    #         3,
    #     ), f"ESKF.innovation_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
    #     assert R_GNSS.shape == (
    #         3,
    #         3,
    #     ), f"ESKF.innovation_GNSS: R_GNSS shape incorrect {R_GNSS.shape}"
    #     assert lever_arm.shape == (
    #         3,
    #     ), f"ESKF.innovation_GNSS: lever_arm shape incorrect {lever_arm.shape}"

    #     # H = np.block([np.eye(3), np.zeros((3, 12))])  # measurement matrix
    #     H = self.get_measurement_jacobian(x_nominal) 
    #     Hx = np.eye(3, M=16)  #Jacobian for standard GNSS measurements (10.80)

    #     #eq 10.75 v = z - h(x)
    #     v = z_GNSS_position - Hx @ x_nominal  # Innovation

    #     # leverarm compensation
    #     if not np.allclose(lever_arm, 0):
    #         R = quaternion_to_rotation_matrix(
    #             x_nominal[ATT_IDX], debug=self.debug)
    #         H[:, ERR_ATT_IDX] = - \
    #             R @ cross_product_matrix(lever_arm, debug=self.debug)
    #         v -= R @ lever_arm
        
        
    #     #eq 10.75
    #     S = H @ P @ H.T + R_GNSS  # innovation covariance

    #     assert v.shape == (
    #         3,), f"ESKF.innovation_GNSS: v shape incorrect {v.shape}"
    #     assert S.shape == (
    #         3, 3), f"ESKF.innovation_GNSS: S shape incorrect {S.shape}"
    #     return v, S