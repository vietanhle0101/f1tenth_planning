from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.config.dynamics_config import dynamics_config

import numpy as np

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

    def f(self, state: dict, control: np.ndarray, params: dynamics_config = None) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input.

        Args:
            state (dict): observation as returned from the environment.
            control (np.ndarray): control input as (steering_angle, speed)
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        if params is not None:
            self.params = params

        x, y, delta, v, yaw = state['x'], state['y'], state['delta'], state['v'], state['yaw']
        delta_v, a = control

        # Compute the state derivative
        dx = v * np.cos(yaw)
        dy = v * np.sin(yaw)
        ddelta = delta_v
        dv = a
        dyaw = (v / self.params.WHEELBASE) * np.tan(delta) 

        return np.array([dx, dy, ddelta, dv, dyaw])