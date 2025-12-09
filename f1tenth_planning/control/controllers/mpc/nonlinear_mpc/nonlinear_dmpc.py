from f1tenth_planning.control.controllers.mpc.mpc import MPC_Controller
from f1tenth_planning.control.config.controller_config import (
    mpc_config,
    dynamic_mpc_config,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import Dynamic_Bicycle_Model
from f1tenth_planning.control.solvers.nonlinear_mpc_solver import Nonlinear_MPC_Solver
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_gym.envs.track import Track
import numpy as np


class Nonlinear_Dynamic_MPC_Planner(MPC_Controller):
    def __init__(
        self,
        track: Track,
        params: dynamics_config = None,
        model: Dynamic_Bicycle_Model = None,
        config: mpc_config = None,
        solver: Nonlinear_MPC_Solver = None,
    ):
        """
        Convenience class that uses Nonlinear MPC solver with dynamic bicycle model.

        Args:
            track (f1tenth_gym_ros:Track): track object, contains the reference raceline
            config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
            params (dynamics_config, optional): Vehicle parameters for the dynamic model. If none,
            default f1tenth_params() will be used.
        """
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = Dynamic_Bicycle_Model(params)
        if config is None:
            config = dynamic_mpc_config()
            # x = [x, y, delta, v, yaw, yaw_rate, beta]
            config.x_min = np.array(
                [
                    -np.inf,
                    -np.inf,
                    params.MIN_STEER,
                    params.MIN_SPEED,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ]
            )
            config.x_max = np.array(
                [
                    np.inf,
                    np.inf,
                    params.MAX_STEER,
                    params.MAX_SPEED,
                    np.inf,
                    np.inf,
                    np.inf,
                ]
            )
            # u = [delta_v, a]
            config.u_min = np.array(
                [
                    params.MIN_DSTEER,
                    params.MIN_ACCEL,
                ]
            )
            config.u_max = np.array(
                [
                    params.MAX_DSTEER,
                    params.MAX_ACCEL,
                ]
            )
        if solver is None:
            solver = Nonlinear_MPC_Solver(config=config, model=model)
        super(Nonlinear_Dynamic_MPC_Planner, self).__init__(
            track,
            solver,
            model,
            params,
        )
