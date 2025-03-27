from dataclasses import dataclass, field
import numpy as np


@dataclass
class mpc_config:
    """
    Configuration for the MPC controller. Includes the following parameters:

    Args:
        nx (int): Number of states.
        nu (int): Number of control inputs.
        N (int): Planning horizon for the MPC controller.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control input cost matrix.
        Rd (np.ndarray): Control input derivative cost matrix (action rate cost).
        P (np.ndarray): Terminal cost matrix.
        dt (float): Time discretization interval.
    """
    nx: int
    nu: int
    N: int
    Q: np.ndarray
    R: np.ndarray
    Rd: np.ndarray
    P: np.ndarray
    dt: float

    x_min: np.ndarray = field(default=None)
    x_max: np.ndarray = field(default=None)
    u_min: np.ndarray = field(default=None)
    u_max: np.ndarray = field(default=None)
    ud_min: np.ndarray = field(default=None)
    ud_max: np.ndarray = field(default=None)
  
def kinematic_mpc_config():
    # [x, y, delta, v, yaw]
    return mpc_config(
        nx=5,
        nu=2,
        N=10,
        Q=np.diag([18.0, 18.0, 0.0, 1.2, 18.0]),
        R=np.diag([0.01, 0.4]),
        Rd=np.diag([0.002, 0.01]),
        P=np.diag([18.0, 18.0, 0.0, 1.2, 18.0]),
        dt=0.1
    )
    
def dynamic_mpc_config():
    # [x, y, delta, v, yaw, yaw_rate, beta]
    return mpc_config(
        nx=7,
        nu=2,
        N=5,
        Q=np.diag([18.5, 18.5, 0.0, 7.5, 1.5, 0.4, 0.0]),
        R=np.diag([0.3, 3.9]),
        Rd=np.diag([0.3, 3.9]),
        P=np.diag([18.5, 18.5, 0.0, 7.5, 1.5, 0.4, 0.0]),
        dt=0.1
    )


@dataclass
class lqr_config:
    """
    Configuration for the LQR controller. Includes the following parameters:

    Args:
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control input cost matrix."
        max_iterations (int): Maximum number of iterations for the LQR solver.
        eps (float): Tolerance for convergence.
        dt (float): Time discretization interval.
    """
    Q: np.ndarray = field(default=None)
    R: np.ndarray = field(default=None)
    max_iterations: int = None
    eps: float = None
    dt: float = None

    def __post_init__(self):
        self.Q = np.diag([0.999, 0.0, 0.0066, 0.0])
        self.R = np.array([[0.75]])
        self.max_iterations = 50
        self.eps = 0.01
        self.dt = 0.01