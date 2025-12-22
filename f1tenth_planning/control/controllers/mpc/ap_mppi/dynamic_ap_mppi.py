from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.config.controller_config import (
    MPCConfig,
    dynamic_mppi_config,
)
from f1tenth_planning.control.dynamics_model import DynamicsModel
from f1tenth_planning.control.solvers import APMPPISolver


class DynamicAPMPPIPlanner(MPCController):
    """
    Convenience class that uses MPPI solver with dynamic bicycle model.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        solver (APMPPISolver, optional): APMPPISolver solver object, contains APMPPISolver parameters
        model (DynamicsModel, optional): dynamics model object, contains the vehicle dynamics
        params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none,
        config (MPCConfig, optional): MPC configuration object, contains MPC costs and constraints
    """

    def __init__(
        self,
        track: Track,
        params: DynamicsConfig = None,
        model: DynamicsModel = None,
        config: MPCConfig = None,
        solver: APMPPISolver = None,
        pre_processing_fn=None,
    ):
        print("Initiailizing Dynamic MPPI Planner (convenience class)")
        if not isinstance(solver, APMPPISolver) and solver is not None:
            raise ValueError("Solver must be an instance of MPPISolver")
        if not isinstance(model, DynamicsModel) and model is not None:
            raise ValueError("Model must be an instance of DynamicsModel")
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = DynamicBicycleModel(params)
        if config is None:
            config = dynamic_mppi_config()
        if solver is None:
            solver = APMPPISolver(config, model)
        super(DynamicAPMPPIPlanner, self).__init__(
            track,
            solver,
            model,
            params,
            pre_processing_fn,
        )
