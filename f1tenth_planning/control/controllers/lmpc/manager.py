"""
LMPC iteration manager: collects rollouts and triggers safe-set/value updates at iteration end.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from f1tenth_planning.control.controllers.lmpc.components import (
    SafeSetStore,
    ValueFunctionModel,
    EpisodeLogger,
)
from f1tenth_planning.control.config.controller_config import LMPCConfig


class LMPCIterationManager:
    def __init__(
        self,
        safe_set_store: SafeSetStore,
        value_function: Optional[ValueFunctionModel],
        config: LMPCConfig,
        logger: Optional[EpisodeLogger] = None,
    ):
        self.safe_set_store = safe_set_store
        self.value_function = value_function
        self.config = config
        self.logger = logger or EpisodeLogger()
        self.reset_rollout()

    def reset_rollout(self):
        self.states: list[np.ndarray] = []
        self.controls: list[np.ndarray] = []

    def record_step(self, state: np.ndarray, control: np.ndarray):
        self.states.append(state.copy())
        self.controls.append(control.copy())

    def complete_iteration(self, info: Optional[dict] = None):
        if len(self.states) == 0:
            return
        states_arr = np.vstack(self.states)
        controls_arr = np.vstack(self.controls).T if len(self.controls) > 0 else np.zeros((states_arr.shape[1], 0))
        self.safe_set_store.add_trajectory(states_arr, controls_arr)
        if self.value_function is not None:
            self.value_function.update(self.safe_set_store)
        self.logger.append(states_arr, controls_arr, info or {})
        self.reset_rollout()
