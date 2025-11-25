from f1tenth_planning.control.controllers.lmpc.base import LMPCController
from f1tenth_planning.control.controllers.lmpc.components import (
    SimpleSafeSetStore,
    SimpleValueFunctionModel,
)
from f1tenth_planning.control.controllers.lmpc.manager import LMPCIterationManager
from f1tenth_planning.control.solvers import SafeMPPISolver
from f1tenth_planning.control.config.controller_config import (
    lmpc_config,
    safe_mppi_config,
)
from f1tenth_planning.control.config.dynamics_config import f1tenth_params


class SITLMPCPlanner(LMPCController):
    """
    Convenience planner wiring LMPCController with the Safe-MPPI stub components.
    """

    def __init__(
        self,
        track,
        params=None,
        lmpc_cfg: lmpc_config = None,
        safe_mppi_cfg: safe_mppi_config = None,
        base_controller=None,
    ):
        lmpc_cfg = lmpc_cfg or lmpc_config()
        safe_mppi_cfg = safe_mppi_cfg or safe_mppi_config()
        solver = SafeMPPISolver(safe_mppi_cfg)
        params = params or (
            track.vehicle_params
            if hasattr(track, "vehicle_params")
            else f1tenth_params()
        )
        super().__init__(
            track,
            solver,
            params=params,
            lmpc_cfg=lmpc_cfg,
            base_controller=base_controller,
        )


__all__ = [
    "LMPCController",
    "ITLMPCPlanner",
    "SimpleSafeSetStore",
    "SimpleValueFunctionModel",
    "LMPCIterationManager",
]
