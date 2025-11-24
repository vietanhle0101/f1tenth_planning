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
        if solver is None:
            solver = MPPI_Solver(config, model)
        super(Dynamic_MPPI_Planner, self).__init__(
            track,
            solver,
            model,
            params,
            pre_processing_fn,
        )
