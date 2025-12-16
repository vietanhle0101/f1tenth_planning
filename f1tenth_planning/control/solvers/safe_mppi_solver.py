"""
Safe-MPPI solver stub implementing the LMPCSolver interface.

This keeps the LMPCController API satisfied while leaving room to plug in the
full LMPPI_jax-backed Safe-MPPI implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from f1tenth_planning.control.controllers.lmpc.components import (
    LMPCSolver,
    ValueFunctionModel,
)
from f1tenth_planning.control.config.controller_config import APMPPIConfig


class APMPPISolver(LMPCSolver):
    def __init__(self, config: APMPPIConfig):
        self.config = config

    def solve(
        self,
        x0: np.ndarray,
        ref_traj: Optional[np.ndarray],
        safe_set: Optional[Dict[str, Any]],
        value_fn: Optional[ValueFunctionModel],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Placeholder implementation: returns zero controls and hold-state predictions.
        Replace with LMPPI_jax-backed solve when available.
        """
        N = self.config.N
        nx = self.config.nx
        nu = self.config.nu
        x_pred = np.tile(np.array(x0).reshape(nx, 1), (1, N + 1))
        u_pred = np.zeros((nu, N))
        meta = {
            "status": "stub",
            "safe_set_used": safe_set is not None and safe_set.get("states") is not None,
        }
        return x_pred, u_pred, meta
