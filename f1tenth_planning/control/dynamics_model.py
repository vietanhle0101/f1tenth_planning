from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
import casadi as ca
from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.config.dynamics_config import dynamics_config

class Dynamics_Model(ABC):
    @abstractmethod
    def __init__(self, track: Track, params: dynamics_config) -> None:
        """
        Initialize the dynamics model.

        Args:
            track (Track): track object with raceline, may be used for frenet coordinates
            params (dynamics_config): vehicle dynamics parameters
        """
        self.track = track
        self.params = params

    @abstractmethod
    def f(self, state: np.ndarray, control: np.ndarray, params: dynamics_config = None) -> np.ndarray:
        """
        (Non-)linear dynamics model. This function computes the state derivative given the current state and control input. Should be 
        paired with a numerical integrator to propagate the state forward in time (e.g. Runge-Kutta, Euler). All noise in state and control
        should be handled externally.
        
        Mathematically:
            \dot{x} = f(x, u)

        Args:
            state (np.ndarray): observation as returned from the environment.
            control (np.ndarray): control input as (steering_angle, speed)
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        raise NotImplementedError("control method not implemented")

    @abstractmethod
    def f_casadi(self, params: dynamics_config = None) -> ca.Function:
        """
        (Non-)linear dynamics model in CasADi symbolic form. This function computes the state derivative given the current state and control 
        input. Should be paired with a numerical integrator to propagate the state forward in time (e.g. Runge-Kutta, Euler). All noise in state 
        and control should be handled externally. This function will create symbolic variables for each state and control input, and return a
        CasADi function that can be used to compute the state derivative.
        
        Mathematically:
            \dot{x} = f(x, u)
        
        Args:
            params (dynamics_config): vehicle dynamics parameters, overwrites self.params if not None

        Returns:
            ca.Function: CasADi function for the state derivative
        """
        raise NotImplementedError("control method not implemented")
    
    @abstractmethod
    def linearize_around_state(self, state: np.ndarray, control: np.ndarray, params: dynamics_config = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Linearize the dynamics model around a given state and control input. This function computes the state Jacobian and control Jacobian
        at the given state and control input. These Jacobians can be used in model-based controllers.

        Mathematically:
            A = df_dx(x, u)
            B = df_du(x, u)

        Args:
            state (np.ndarray): observation as returned from the environment.
            control (np.ndarray): control input as (steering_angle, speed)
            params (dynamics_config): vehicle dynamics parameters, overwrites self.params if not None

        Returns:
            tuple[np.ndarray, np.ndarray]: state Jacobian, control Jacobian
        """
        raise NotImplementedError("linearize_around_state method not implemented")
    
    @property
    def params(self) -> dynamics_config:
        """
        Get the dynamics configuration parameters.
        """
        return self._params

    @params.setter
    def params(self, value: dynamics_config) -> None:
        """
        Set the dynamics configuration parameters without updating nx and nu fields.
        """
        assert isinstance(value, dynamics_config), f"Expected dynamics_config, got {type(value)}"
        self._params = value

