from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.config.dynamics_config import dynamics_config

import numpy as np
import casadi as ca

class Kinematic_Model(Dynamics_Model):
    """
    Kinematic bicycle model for vehicle dynamics. 

    Vehicle State: 
        - [x, y, delta, v, yaw]
    Control:
        - [delta_v, a]
    
    Reference:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads

    Args:
        dynamics_config: dynamics_config - vehicle dynamics configuration
    """
    def __init__(self, track: Track, params: dynamics_config):
        super().__init__(track, params)     
        self.nx = 5
        self.nu = 2   

    def f(self, state: dict, control: np.ndarray, params: dynamics_config = None) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input.

        Args:
            state (np.ndarray): dynamic state as [x, y, delta, v, yaw]
            control (np.ndarray): control input as (steering_velocity, acceleration)
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        if params is not None:
            self.params = params

        x, y, delta, v, yaw = state
        delta_v, a = control

        # Compute the state derivative
        dx = v * np.cos(yaw)
        dy = v * np.sin(yaw)
        ddelta = delta_v
        dv = a
        dyaw = (v / self.params.WHEELBASE) * np.tan(delta) 

        return np.array([dx, dy, ddelta, dv, dyaw])
    
    def f_casadi(self, params: dynamics_config = None) -> ca.Function:
        if params is not None:
            self.params = params

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        delta = ca.SX.sym('delta')
        v = ca.SX.sym('v')
        yaw = ca.SX.sym('yaw')
        states = ca.vertcat(
            x,
            y,
            delta,
            v,
            yaw
        )
        # control symbolic variables
        a = ca.SX.sym('a')
        delta_v = ca.SX.sym('delta_v')
        controls = ca.vertcat(
            delta_v,
            a
        )
        RHS = ca.vertcat(
                            v * ca.cos(yaw),  # dx/dt = v * cos(yaw)
                            v * ca.sin(yaw),  # dy/dt = v * sin(yaw)
                            delta_v,  # d(delta)/dt = delta_v
                            a,  # dv/dt = a
                            (v/(self.params.WHEELBASE)) * ca.tan(delta)  # dyaw/dt = (v/(Lx+Ly)) * tan(delta)
                        ) # dx/dt = f(x,u)

        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        f = ca.Function('f', [states, controls], [RHS])
        return f

    def linearize_around_state(self, state: np.ndarray, control: np.ndarray, params: dynamics_config = None) -> tuple[np.ndarray, np.ndarray]:
        x, y, delta, v, yaw = state

        # State (or system) matrix A, 5x5
        A = np.zeros((self.nx, self.nx))

        A[0, 3] = np.cos(yaw)                                             # dx/d(v)
        A[0, 4] = -v * np.sin(yaw)                                        # dx/d(yaw)

        A[1, 3] = np.sin(yaw)                                             # dy/d(v)
        A[1, 4] = v * np.cos(yaw)                                         # dy/d(yaw)

        A[4, 2] = (v / self.params.WHEELBASE) * (1 / np.cos(delta) ** 2)  # dyaw/d(delta)
        A[4, 3] = np.tan(delta) / self.params.WHEELBASE                   # dyaw/d(v)

        # Input Matrix B; 5x2
        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = 1

        return A, B