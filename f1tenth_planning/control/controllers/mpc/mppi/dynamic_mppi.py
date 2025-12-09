from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.controllers.mpc.mpc import MPC_Controller
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import Dynamic_Bicycle_Model
from f1tenth_planning.control.config.controller_config import (
    mpc_config,
    dynamic_mppi_config,
)
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.solvers import MPPI_Solver
import jax.numpy as jnp


class Dynamic_MPPI_Planner(MPC_Controller):
    """
    Convenience class that uses MPPI solver with dynamic bicycle model.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        solver (MPPI_Solver, optional): MPPI solver object, contains MPPI parameters
        model (Dynamics_Model, optional): dynamics model object, contains the vehicle dynamics
        params (dynamics_config, optional): Vehicle parameters for the dynamic model. If none,
        config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        params: dynamics_config = None,
        model: Dynamics_Model = None,
        config: mpc_config = None,
        solver: MPPI_Solver = None,
        pre_processing_fn=None,
    ):
        print("Initiailizing Dynamic MPPI Planner (convenience class)")
        if not isinstance(solver, MPPI_Solver) and solver is not None:
            raise ValueError("Solver must be an instance of MPPI_Solver")
        if not isinstance(model, Dynamics_Model) and model is not None:
            raise ValueError("Model must be an instance of Dynamics_Model")
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = Dynamic_Bicycle_Model(params)
        if config is None:
            config = dynamic_mppi_config()
            # x = [x, y, delta, v, yaw, yaw_rate, beta]
            config.x_min = jnp.array(
                [
                    -jnp.inf,
                    -jnp.inf,
                    params.MIN_STEER,
                    params.MIN_SPEED,
                    -jnp.inf,
                    -jnp.inf,
                    -jnp.inf,
                ]
            )
            config.x_max = jnp.array(
                [
                    jnp.inf,
                    jnp.inf,
                    params.MAX_STEER,
                    params.MAX_SPEED,
                    jnp.inf,
                    jnp.inf,
                    jnp.inf,
                ]
            )
            # u = [delta_v, a]
            config.u_min = jnp.array(
                [
                    params.MIN_DSTEER,
                    params.MIN_ACCEL,
                ]
            )
            config.u_max = jnp.array(
                [
                    params.MAX_DSTEER,
                    params.MAX_ACCEL,
                ]
            )
        if solver is None:
            solver = MPPI_Solver(config, model)
        super(Dynamic_MPPI_Planner, self).__init__(
            track,
            solver,
            model,
            params,
            pre_processing_fn,
        )
        self.sampled_trajectories_render = None

    def render_sampled_trajectories(self, e):
        """
        Render the sampled trajectories on the environment.

        Args:
            e: The environment renderer instance used for drawing.
        """
        # Render the sampled trajectories
        ORANGE = (255, 165, 0)
        if self.solver.samples is not None:
            sampled_trajectories = self.solver.samples[1]
            if self.sampled_trajectories_render is None:
                self.sampled_trajectories_render = [None] * len(
                    sampled_trajectories // 100
                )
            for idx in range(0, len(sampled_trajectories), 100):
                sampled_xy = sampled_trajectories[idx][:, :2]
                if self.sampled_trajectories_render[idx] is None:
                    self.sampled_trajectories_render[idx] = e.render_lines(
                        sampled_xy, color=ORANGE, size=4
                    )
                else:
                    self.sampled_trajectories_render[idx].setData(
                        sampled_xy[:, 0], sampled_xy[:, 1]
                    )
