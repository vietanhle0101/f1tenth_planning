from __future__ import annotations

import numpy as np
from typing import Optional

from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum

from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.config.controller_config import LMPCConfig
from f1tenth_planning.control.controllers.lmpc.components import (
    LMPCSolver,
    SafeSetStore,
    ValueFunctionModel,
    SimpleSafeSetStore,
    SimpleValueFunctionModel,
)
from f1tenth_planning.control.controllers.lmpc.manager import LMPCIterationManager


class LMPCController(Controller):
    """
    High-level LMPC controller skeleton.
    """

    def __init__(
        self,
        track: Track,
        solver: LMPCSolver,
        safe_set_store: Optional[SafeSetStore] = None,
        value_function: Optional[ValueFunctionModel] = None,
        iteration_manager: Optional[LMPCIterationManager] = None,
        base_controller: Optional[Controller] = None,
        params: DynamicsConfig = f1tenth_params(),
        lmpc_cfg: Optional[LMPCConfig] = None,
    ):
        super().__init__(
            track,
            params,
            control_mode=(SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl),
        )
        self.waypoints = np.vstack(
            [
                track.raceline.xs,
                track.raceline.ys,
                np.zeros_like(track.raceline.xs),
                track.raceline.vxs,
                track.raceline.yaws,
                np.zeros_like(track.raceline.xs),
                np.zeros_like(track.raceline.xs),
            ]
        ).T

        self.solver = solver
        self.safe_set_store = safe_set_store or SimpleSafeSetStore()
        self.value_function = value_function or SimpleValueFunctionModel()
        self.lmpc_cfg = lmpc_cfg or LMPCConfig()
        self.base_controller = base_controller
        self.iteration_manager = iteration_manager or LMPCIterationManager(
            self.safe_set_store, self.value_function, self.lmpc_cfg
        )

        self.x_pred = None
        self.u_pred = None
        self.ref_traj = None
        self.local_plan = None
        self.control_solution = None
        self.mpc_solution_render = None
        self.local_plan_render = None

    def plan(self, state: dict, waypoints=None, **kwargs):
        if waypoints is not None:
            self.waypoints = waypoints

        x = state["pose_x"]
        y = state["pose_y"]
        v = state["linear_vel_x"]
        yaw = state["pose_theta"]
        x0 = np.array([x, y, state["delta"], v, yaw, state["ang_vel_z"], state["beta"]])

        # If safe set is not yet populated enough, use the base controller if provided.
        num_traj = getattr(self.safe_set_store, "num_trajectories", 0)
        if num_traj < self.lmpc_cfg.ss_size and self.base_controller is not None:
            control = self.base_controller.plan(state, waypoints=self.waypoints)
            self.iteration_manager.record_step(x0, np.array(control))
            return np.array(control)

        safe_set_snapshot = self.safe_set_store.snapshot()

        x_pred, u_pred, meta = self.solver.solve(
            x0, None, safe_set_snapshot, self.value_function, **kwargs
        )
        self.x_pred = x_pred
        self.u_pred = u_pred
        self.local_plan = None
        self.control_solution = (
            np.array(x_pred[:2, :]) if x_pred is not None else None
        )

        if u_pred is None or u_pred.shape[1] == 0:
            control = np.zeros((2,))
        else:
            control = np.array(u_pred[:, 0]).flatten()

        self.iteration_manager.record_step(x0, control)
        return control

    def render_control_solution(self, e):
        if self.x_pred is not None:
            if self.mpc_solution_render is None:
                self.mpc_solution_render = e.render_points(
                    self.control_solution.T, color=(128, 0, 0), size=4
                )
            else:
                self.mpc_solution_render.setData(self.control_solution.T)

    def render_local_plan(self, e):
        if self.local_plan is not None:
            if self.local_plan_render is None:
                self.local_plan_render = e.render_closed_lines(
                    self.local_plan, color=(0, 0, 128), size=4
                )
            else:
                self.local_plan_render.setData(self.local_plan)

    def complete_iteration(self, info: Optional[dict] = None):
        self.iteration_manager.complete_iteration(info)
