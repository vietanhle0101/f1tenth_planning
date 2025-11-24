from f1tenth_planning.control.controllers.mpc.mpc import MPC_Controller
from f1tenth_planning.control.config.controller_config import (
    mpc_config,
    kinematic_mpc_config,
)
from f1tenth_planning.control.dynamics_models.kinematic_model import (
    Kinematic_Bicycle_Model,
    _extract_kinematic_state,
)
from f1tenth_planning.control.solvers.LTV_mpc_solver import LTV_MPC_Solver
from f1tenth_planning.control.config.dynamics_config import (
    dynamics_config,
    f1tenth_params,
)
from f1tenth_gym.envs.track import Track


class Kinematic_MPC_Planner(MPC_Controller):
    def __init__(
        self,
        track: Track,
        params: dynamics_config = None,
        model: Kinematic_Bicycle_Model = None,
        config: mpc_config = None,
        solver: LTV_MPC_Solver = None,
        pre_processing_fn=_extract_kinematic_state,
    ):
        """
        Convenience class that uses LTV MPC solver with kinematic bicycle model.

        Args:
            track (f1tenth_gym_ros:Track): track object, contains the reference raceline
            config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
            params (dynamics_config, optional): Vehicle parameters for the kinematic model. If none,
            default f1tenth_params() will be used.
        """
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = Kinematic_Bicycle_Model(params)
        if config is None:
            config = kinematic_mpc_config()
        if solver is None:
            solver = LTV_MPC_Solver(config=config, model=model)
        super(Kinematic_MPC_Planner, self).__init__(
            track=track,
            solver=solver,
            model=model,
            params=params,
            pre_processing_fn=pre_processing_fn,
        )
