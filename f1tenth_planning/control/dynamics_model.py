from __future__ import annotations
from abc import abstractmethod, ABC

import jax
from jax import numpy as jnp
import numpy as np
import casadi as ca
from f1tenth_planning.control.config.dynamics_config import dynamics_config

class Dynamics_Model(ABC):
    @abstractmethod
    def __init__(self, params: dynamics_config) -> None:
        """
        Initialize the dynamics model.

        Args:
            params (dynamics_config): vehicle dynamics parameters
        """
        self.params = params

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
    
    def f_casadi_opti(self, state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX:
        """
        Casadi OptiStack compatible function for the dynamic model. 
        
        Args:
            x (ca.SX): (nx, 1) state vector
            u (ca.SX): (nu, 1) control vector
            p (ca.SX): (num_p, 1) parameter vector
        Returns:
            ca.SX: (nx, 1) state derivative
        """
        raise NotImplementedError("control method not implemented")

    def f_jax(self, state: jnp.ndarray, control: jnp.ndarray, params: jnp.ndarray = None) -> jnp.ndarray:
        """
        (Non-)linear dynamics model in JAX. This function computes the state derivative given the current state and control input. Should be 
        paired with a numerical integrator to propagate the state forward in time (e.g. Runge-Kutta, Euler). All noise in state and control
        should be handled externally.
        
        Mathematically:
            \dot{x} = f(x, u)

        Args:
            state (jnp.ndarray): observation as returned from the environment.
            control (jnp.ndarray): control input as (steering_angle, speed)
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            jnp.ndarray: state derivative
        """
        raise NotImplementedError("control method not implemented")
    
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
    
    def parameters_vector_from_config(self, params: dynamics_config) -> np.ndarray:
        """
        Convert the dynamics configuration parameters into a vector format. This function is useful for optimization problems where the
        parameters need to be passed as a vector.

        Args:
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            np.ndarray: (num_params, 1) vector of parameters
        """
        raise NotImplementedError("parameters_vector_from_config method not implemented")
    
    def config_from_parameters_vector(self, params: np.ndarray) -> dynamics_config:
        """
        Convert a vector of parameters into a dynamics configuration object. This function is useful for optimization problems where the
        parameters need to be passed as a vector.

        Args:
            params (np.ndarray): (num_params, 1) vector of parameters

        Returns:
            dynamics_config: vehicle dynamics parameters
        """
        raise NotImplementedError("config_from_parameters_vector method not implemented")

    @property
    def num_params(self) -> dynamics_config:
        """
        Get the number of parameters from dynamics_config that are actively used in the model.
        This is useful for determining the size of the parameter vector in optimization problems.
        """
        raise NotImplementedError("num_params method not implemented")

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

