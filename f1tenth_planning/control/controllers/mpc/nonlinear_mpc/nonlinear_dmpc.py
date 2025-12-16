from f1tenth_planning.control.controllers.mpc.mpc import MPCController
from f1tenth_planning.control.config.controller_config import (
    MPCConfig,
    dynamic_mpc_config,
)
from f1tenth_planning.control.dynamics_models.dynamic_model import DynamicBicycleModel
from f1tenth_planning.control.solvers.nonlinear_mpc_solver import NonlinearMPCSolver
from f1tenth_planning.control.config.dynamics_config import (
    DynamicsConfig,
    f1tenth_params,
)
from f1tenth_gym.envs.track import Track


class NonlinearDynamicMPCPlanner(MPCController):
    def __init__(
        self,
        track: Track,
        params: DynamicsConfig = None,
        model: DynamicBicycleModel = None,
        config: MPCConfig = None,
        solver: NonlinearMPCSolver = None,
    ):
        """
        Convenience class that uses Nonlinear MPC solver with dynamic bicycle model.

        Args:
            track (f1tenth_gym_ros:Track): track object, contains the reference raceline
            config (MPCConfig, optional): MPC configuration object, contains MPC costs and constraints
            params (DynamicsConfig, optional): Vehicle parameters for the dynamic model. If none,
            default f1tenth_params() will be used.
        """
        if params is None:
            params = f1tenth_params()
        if model is None:
            model = DynamicBicycleModel(params)
        if config is None:
            config = dynamic_mpc_config()
        if solver is None:
            solver = NonlinearMPCSolver(config=config, model=model)
        super(NonlinearDynamicMPCPlanner, self).__init__(
            track,
            solver,
            model,
            params,
        )
