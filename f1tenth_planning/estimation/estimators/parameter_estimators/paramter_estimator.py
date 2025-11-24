from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_planning.control.config.dynamics_config import dynamics_config
from f1tenth_planning.control.dynamics_model import Dynamics_Model


class ParameterEstimator(ABC):
    @abstractmethod
    def __init__(self, initial_params: dynamics_config, model: Dynamics_Model):
        """
        Initialize parameter estimator.

        Args:
            initial_params (dynamics_config): initial guess of the parameters for the estimator.
            model (Dynamics_Model): dynamics model to use for estimation.
        """
        self.params = initial_params
        self.model = model

    @abstractmethod
    def estiamte(
        self,
        state: dict,
        control: np.ndarray,
        new_param_guess: dynamics_config = None,
        **kwargs,
    ) -> dynamics_config:
        """
        Estiamte the parameters of the dynamics model based on the current state and control input. This may use a history of previous states and controls, but internal to the estimator.

        Args:
            state (dict): current state of the system.
            control (np.ndarray): control input applied to the system.
            new_param_guess (dynamics_config, optional): new parameter guess to update the estimator with.
            **kwargs: additional arguments for the estimator.

        Returns:
            dynamics_config: updated parameters of the dynamics model.
        """
        raise NotImplementedError("control method not implemented")
