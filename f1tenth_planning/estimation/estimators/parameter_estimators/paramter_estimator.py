from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
from f1tenth_planning.control.config.dynamics_config import DynamicsConfig
from f1tenth_planning.control.dynamics_model import DynamicsModel


class ParameterEstimator(ABC):
    @abstractmethod
    def __init__(self, initial_params: DynamicsConfig, model: DynamicsModel):
        """
        Initialize parameter estimator.

        Args:
            initial_params (DynamicsConfig): initial guess of the parameters for the estimator.
            model (DynamicsModel): dynamics model to use for estimation.
        """
        self.params = initial_params
        self.model = model

    @abstractmethod
    def estiamte(
        self,
        state: dict,
        control: np.ndarray,
        new_param_guess: DynamicsConfig = None,
        **kwargs,
    ) -> DynamicsConfig:
        """
        Estiamte the parameters of the dynamics model based on the current state and control input. This may use a history of previous states and controls, but internal to the estimator.

        Args:
            state (dict): current state of the system.
            control (np.ndarray): control input applied to the system.
            new_param_guess (DynamicsConfig, optional): new parameter guess to update the estimator with.
            **kwargs: additional arguments for the estimator.

        Returns:
            DynamicsConfig: updated parameters of the dynamics model.
        """
        raise NotImplementedError("control method not implemented")
