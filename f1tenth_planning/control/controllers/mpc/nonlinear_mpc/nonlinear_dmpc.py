from f1tenth_planning.control.controllers.mpc.mpc import MPC_Controller
from f1tenth_planning.control.config.controller_config import (
    mpc_config,
    kinematic_mpc_config,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import Dynamic_Bicycle_Model
from f1tenth_planning.control.solvers.nonlinear_mpc_solver import Nonlinear_MPC_Solver
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_gym.envs.track import Track


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
            config = kinematic_mpc_config()
        if solver is None:
            solver = Nonlinear_MPC_Solver(model, config)
        super(Nonlinear_Dynamic_MPC_Planner, self).__init__(
            track,
            solver,
            model,
            params,
        )
