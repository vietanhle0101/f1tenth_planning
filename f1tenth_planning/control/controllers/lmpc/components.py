"""
LMPC support component interfaces and simple defaults.

These interfaces keep solver/value/safe-set pieces swappable (Safe-MPPI, NF/BNN value nets, etc.)
while letting the controller drive the high-level loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ReferenceGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        x0: np.ndarray,
        waypoints: np.ndarray,
        dt: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Produce a horizon-length reference trajectory aligned with the controller state vector.
        Returns array of shape (horizon + 1, nx).
        """
        raise NotImplementedError


class SafeSetStore(ABC):
    @abstractmethod
    def add_trajectory(self, states: np.ndarray, controls: np.ndarray) -> None:
        """
        Add a completed rollout (states shape (T, nx), controls shape (nu, T-1 or T)).
        Should also compute and store cost-to-go/value targets.
        """
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a lightweight snapshot usable by solvers (e.g., safe set arrays, costs).
        """
        raise NotImplementedError


class ValueFunctionModel(ABC):
    @abstractmethod
    def predict(self, states: np.ndarray) -> np.ndarray:
        """
        Predict value/cost-to-go for provided states (shape (N, nx) or (N, value_dim)).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, safe_set: SafeSetStore) -> None:
        """
        Retrain or refresh the value model using data from the safe set.
        """
        raise NotImplementedError


class LMPCSolver(ABC):
    @abstractmethod
    def solve(
        self,
        x0: np.ndarray,
        ref_traj: Optional[np.ndarray],
        safe_set: Optional[Dict[str, Any]],
        value_fn: Optional[ValueFunctionModel],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Compute predicted state/control sequences.
        Returns x_pred (nx, N+1), u_pred (nu, N), metadata dict.
        """
        raise NotImplementedError


class EpisodeLogger:
    """
    Minimal logger to stash recent rollouts; extend/replace as needed.
    """

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.rollouts: list[Dict[str, np.ndarray]] = []

    def append(self, states: np.ndarray, controls: np.ndarray, info: Dict[str, Any]):
        self.rollouts.append({"states": states, "controls": controls, "info": info})
        if len(self.rollouts) > self.max_history:
            self.rollouts.pop(0)


class SimpleSafeSetStore(SafeSetStore):
    """
    Lightweight safe set store that mirrors the LinearLMPC addTrajectory/computeCost pattern.
    Cost-to-go is computed as cumulative quadratic norm over the stored trajectory.
    """

    def __init__(self, max_trajectories: int = 5):
        self.max_trajectories = max_trajectories
        self.states: list[np.ndarray] = []
        self.controls: list[np.ndarray] = []
        self.costs: list[np.ndarray] = []

    def add_trajectory(self, states: np.ndarray, controls: np.ndarray) -> None:
        costs = self._compute_cost(states, controls)
        self.states.append(states.copy())
        self.controls.append(controls.copy())
        self.costs.append(costs)
        if len(self.states) > self.max_trajectories:
            self.states.pop(0)
            self.controls.pop(0)
            self.costs.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        if not self.states:
            return {"states": None, "costs": None}
        flat_states = np.concatenate(self.states, axis=0)
        flat_costs = np.concatenate(self.costs, axis=0)
        return {"states": flat_states, "costs": flat_costs, "count": len(self.states)}

    @property
    def num_trajectories(self) -> int:
        return len(self.states)

    @staticmethod
    def _compute_cost(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        # Backward cost-to-go: last state has zero terminal cost.
        T = states.shape[0]
        costs = np.zeros((T,))
        for i in range(T - 2, -1, -1):
            state_cost = float(np.dot(states[i], states[i]))
            ctrl_cost = float(np.dot(controls[:, min(i, controls.shape[1] - 1)], controls[:, min(i, controls.shape[1] - 1)]))
            costs[i] = state_cost + ctrl_cost + costs[i + 1]
        return costs


class SimpleValueFunctionModel(ValueFunctionModel):
    """
    Stub value model that uses stored cost-to-go targets from the safe set snapshot.
    """

    def __init__(self):
        self.last_snapshot: Optional[Dict[str, Any]] = None

    def predict(self, states: np.ndarray) -> np.ndarray:
        # If no data is available, return zeros.
        if self.last_snapshot is None or self.last_snapshot.get("states") is None:
            return np.zeros((states.shape[0],))
        # Nearest-neighbor on stored states.
        ref_states = self.last_snapshot["states"]
        ref_costs = self.last_snapshot["costs"]
        dists = np.linalg.norm(ref_states[None, :, :] - states[:, None, :], axis=2)
        nn_idx = np.argmin(dists, axis=1)
        return ref_costs[nn_idx]

    def update(self, safe_set: SafeSetStore) -> None:
        self.last_snapshot = safe_set.snapshot()
