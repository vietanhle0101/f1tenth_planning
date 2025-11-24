import numpy as np
from f1tenth_gym.envs.track import Track
from f1tenth_planning.utils.utils import calc_interpolated_reference_trajectory
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.config.controller_config import (
    dynamic_mppi_config,
    mpc_config,
)
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_planning.control.mpc_solver import MPC_Solver
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_gym.envs.action import SteerActionEnum, LongitudinalActionEnum
from f1tenth_planning.utils.utils import jnp_to_np


class MPC_Controller(Controller):
    """
    MPPI Controller, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    All vehicle pose used by the planner should be in the map frame.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        solver: MPC_Solver,
        model: Dynamics_Model,
        params: dynamics_config = f1tenth_params(),
        pre_processing_fn=None,
    ):
        super().__init__(
            track,
            params,
            control_mode=(SteerActionEnum.Steering_Speed, LongitudinalActionEnum.Accl),
        )
        self.waypoints = np.vstack(
            [
                track.raceline.xs,  # x
                track.raceline.ys,  # y
                np.zeros_like(track.raceline.xs),  # steering angle reference
                track.raceline.vxs,  # v
                track.raceline.yaws,  # yaw
                np.zeros_like(track.raceline.xs),  # yaw rate reference
                np.zeros_like(track.raceline.xs),  # slip angle
            ]
        ).T

        self.model = model
        self.solver = solver

        self.pre_processing_fn = pre_processing_fn

        self.x_pred = None
        self.ref_traj = None

        self.control_solution = None
        self.local_plan = None

        self.mpc_solution_render = None
        self.local_plan_render = None

    def render_control_solution(self, e):
        """
        Callback to render the lookahead point on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.x_pred is not None:
            if self.mpc_solution_render is None:
                self.mpc_solution_render = e.render_points(
                    self.control_solution.T, color=(128, 0, 0), size=4
                )
            else:
                self.mpc_solution_render.setData(self.control_solution.T)

    def render_local_plan(self, e):
        """
        Render the local plan (series of waypoints) on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        if self.ref_traj is not None:
            if self.local_plan_render is None:
                self.local_plan_render = e.render_closed_lines(
                    self.local_plan, color=(0, 0, 128), size=4
                )
            else:
                self.local_plan_render.setData(self.local_plan)

    def plan(
        self,
        state: dict,
        waypoints=None,
        params: dynamics_config = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
    ):
        """
        Compute the control input for the vehicle using a Kinematic MPC planner.

        Args:
            state (dict): Dictionary containing the vehicle's state.
            waypoints (numpy.ndarray [N x 5], optional): An array of dynamic waypoints, where each waypoint has
            the format [x, y, delta, velocity, heading]. Overrides the static raceline if provided.
            params (dynamics_config, optional): Vehicle parameters for the dynamic model. If none, uses default.
            Q (np.ndarray, optional): State cost matrix. If none, uses default.
            R (np.ndarray, optional): Control input cost matrix. If none, uses default.

        Returns:
            control: A tuple (steering_vel, acc) representing the computed steering velocity and acceleration.
            If no valid lookahead point is found, returns (0.0, 0.0) after issuing a warning.
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError(
                    "Waypoints need to be a (N x m) numpy array with m >= 3!"
                )
            self.waypoints = waypoints
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )

        if Q is not None:
            if Q.shape != (
                self.solver.config.Q.shape[0],
                self.solver.config.Q.shape[1],
            ):
                raise ValueError(
                    f"Q must be of shape {self.solver.config.Q.shape}, got {Q.shape}"
                )

        if R is not None:
            if R.shape != (
                self.solver.config.R.shape[0],
                self.solver.config.R.shape[1],
            ):
                raise ValueError(
                    f"R must be of shape {self.solver.config.R.shape}, got {R.shape}"
                )

        x = state["pose_x"]
        y = state["pose_y"]
        v = state["linear_vel_x"]
        yaw = state["pose_theta"]
        x0 = np.array([x, y, state["delta"], v, yaw, state["ang_vel_z"], state["beta"]])

        cx = self.waypoints[:, 0]
        cy = self.waypoints[:, 1]
        cv = self.waypoints[:, 3]

        self.ref_traj = calc_interpolated_reference_trajectory(
            x,
            y,
            yaw,
            cx,
            cy,
            cv,
            self.solver.config.dt,
            self.solver.config.N,
            self.waypoints,
        ).T.copy()
        p = None
        if params is not None:
            p = self.model.parameters_vector_from_config(params)
            self.params = params

        if self.pre_processing_fn is not None:
            x0, self.ref_traj = self.pre_processing_fn(x0, self.ref_traj)

        self.x_pred, self.u_pred = self.solver.solve(x0, self.ref_traj.T, p=p, Q=Q, R=R)
        self.x_pred = jnp_to_np(self.x_pred)
        self.u_pred = jnp_to_np(self.u_pred)

        self.local_plan = self.ref_traj[:2].T
        self.control_solution = np.array(self.x_pred[:2, :])

        return np.array(self.u_pred[:, 0]).flatten(), {
            "predicted_state": self.x_pred,
            "predicted_control": self.u_pred,
            "steering_angle": self.x_pred[2, 1],
            "velocity": self.x_pred[3, 1],
        }
