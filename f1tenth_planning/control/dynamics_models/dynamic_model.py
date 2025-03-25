from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.config.dynamics_config import dynamics_config

import numpy as np

class Dynamic_Model(Dynamics_Model):
    """
    Dynamic single-track bicycle model for vehicle dynamics. 

    Vehicle State: 
        - [x, y, delta, v, yaw, yaw_rate, slip_angle]
    Control:
        - [delta_v, a]
    
    Reference:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads

    Args:
        dynamics_config: dynamics_config - vehicle dynamics configuration
    """
    def __init__(self, track: Track, params: dynamics_config):
        super().__init__(track, params)     
        self.nx = 7
        self.nu = 2   

    def f(self, state: dict, control: np.ndarray, params: dynamics_config = None) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input.

        Args:
            state (np.ndarray): dynamic state as [x, y, delta, v, yaw, yaw_rate, slip_angle]
            control (np.ndarray): control input as (steering_velocity, acceleration)
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        if params is not None:
            self.params = params

        x, y, delta, v, yaw, yaw_rate, slip_angle = state
        delta_v, a = control

        # Compute the state derivative
        dx = v * np.cos(yaw + slip_angle)
        dy = v * np.sin(yaw + slip_angle)
        ddelta = delta_v
        dv = a

        dyaw = 0
        ddyaw = 0
        dslip_angle = 0
        if np.abs(v) <= 0.1:
            # derivative of yaw "kinemaitcally"
            dyaw = v * np.cos(slip_angle) / self.params.WHEELBASE * np.tan(delta)

            # derivative of slip angle and yaw rate
            dslip_angle = (self.params.LR * delta_v) / (self.params.WHEELBASE * np.cos(delta) ** 2 * (1 + (np.tan(delta) * self.params.LR / self.params.WHEELBASE) ** 2))
            ddyaw = 1 / self.params.WHEELBASE * (a * np.cos(slip_angle) * np.tan(delta) -
                                v * np.sin(slip_angle) * np.tan(delta) * dslip_angle +
                                v * np.cos(slip_angle) * delta_v / np.cos(delta) ** 2)
        else:
            dyaw = yaw_rate

            # Extract params for more readable equations
            mu = self.params.MU
            m = self.params.M
            I = self.params.I
            lr = self.params.LR
            lf = self.params.LF
            C_Sf = self.params.C_SF
            C_Sr = self.params.C_SR
            h = self.params.H
            g = 9.81

            ddyaw = -mu * m / (v * I * (lr + lf)) * (lf ** 2 * C_Sf * (g * lr - a * h) + lr ** 2 * C_Sr * (g * lf + a * h)) * yaw_rate + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h)) * slip_angle + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta

            dslip_angle = (mu / (v ** 2 * (lr + lf)) * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf) - 1) * yaw_rate - mu / (v * (lr + lf)) * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h)) * slip_angle + mu / (v * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta

        return np.array([dx, dy, ddelta, dv, dyaw, ddyaw, dslip_angle])
    
    def linearize_around_state(self, state: np.ndarray, control: np.ndarray, params: dynamics_config = None) -> tuple[np.ndarray, np.ndarray]:
        x, y, delta, v, yaw, yaw_rate, slip_angle = state
        delta_v, a = control

        # State (or system) matrix A, 7x7
        A = np.zeros((self.nx, self.nx))

        # dx/dstate
        A[0, 3] = np.cos(yaw + slip_angle)                                             # dx/d(v)
        A[0, 4] = -v * np.sin(yaw + slip_angle)                                        # dx/d(yaw)
        A[0, 6] = -v * np.sin(yaw + slip_angle)                                        # dx/d(slip_angle)

        # dy/dstate
        A[1, 3] = np.sin(yaw + slip_angle)                                             # dy/d(v)
        A[1, 4] = v * np.cos(yaw + slip_angle)                                         # dy/d(yaw)
        A[1, 6] = v * np.cos(yaw + slip_angle)                                         # dy/d(slip_angle)

        if(np.abs(v) <= 0.1):
            # dyaw/dstate
            A[4, 2] = v * np.cos(slip_angle) / self.params.WHEELBASE * (1 / np.cos(delta) ** 2) # dyaw/ddelta
            A[4, 3] = np.cos(slip_angle) / self.params.WHEELBASE * np.tan(delta)                # dyaw/dv
            A[4, 6] = -v * np.sin(slip_angle) / self.params.WHEELBASE * np.tan(delta)            # dyaw/dslip_angle

            # ddyaw/dstate (self.params.LR * delta_v) / (self.params.WHEELBASE * np.cos(delta) ** 2 * (1 + (np.tan(delta) * self.params.LR / self.params.WHEELBASE) ** 2))
            A[5, 2] = (self.params.LR * delta_v) / (self.params.WHEELBASE * np.cos(delta) ** 2 * (1 + (np.tan(delta) * self.params.LR / self.params.WHEELBASE) ** 2)) # ddyaw/ddelta

            pass
        else:
            # Extract params for more readable equations
            mu = self.params.MU
            m = self.params.M
            I = self.params.I
            lr = self.params.LR
            lf = self.params.LF
            C_Sf = self.params.C_SF
            C_Sr = self.params.C_SR
            h = self.params.H
            g = 9.81

            # dyaw/dstate
            A[4, 5] = 1

            # ddyaw/dstate

            pass

        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = 1

        return A, B