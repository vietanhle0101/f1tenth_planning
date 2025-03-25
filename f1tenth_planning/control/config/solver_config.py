from dataclasses import dataclass
import numpy as np

@dataclass
class solver_config:
    """
    Solver configuration dataclass.
    This dataclass contains the solver parameters used for control and planning.
    
    Args:
        DT: float - time step [s]
        N: int - prediction horizon [steps]
        nx: int - state dimension
        nu: int - input dimension
        Q: float - state cost matrix
        R: float - input cost matrix
        Rd: float - input rate cost matrix
        Qf: float - final state cost matrix
        x_min: np.ndarray - state lower bounds
        x_max: np.ndarray - state upper bounds
        u_min: np.ndarray - input lower bounds
        u_max: np.ndarray - input upper bounds
        ud_min: np.ndarray - input rate lower bounds
        ud_max: np.ndarray - input rate upper bounds
    """
    DT: float
    N: int

    nx: int
    nu: int

    Q: np.ndarray = None
    R: np.ndarray = None
    Rd: np.ndarray = None
    P: np.ndarray = None

    x_min: np.ndarray = None
    x_max: np.ndarray = None
    u_min: np.ndarray = None
    u_max: np.ndarray = None
    ud_min: np.ndarray = None
    ud_max: np.ndarray = None

    def __post_init__(self):
        # Default x_min, x_max, u_min, u_max to infinities if not provided
        if self.x_min is None:
            self.x_min = -np.inf * np.ones(self.nx)
        if self.x_max is None:
            self.x_max = np.inf * np.ones(self.nx)
        if self.u_min is None:
            self.u_min = -np.inf * np.ones(self.nu)
        if self.u_max is None:
            self.u_max = np.inf * np.ones(self.nu)
        if self.ud_min is None:
            self.ud_min = -np.inf * np.ones(self.nu)
        if self.ud_max is None:
            self.ud_max = np.inf * np.ones(self.nu)

        # Check that the dimensions of nx and nu are consistent with Q, Qf, R
        assert self.Q.shape == (self.nx, self.nx), "Q matrix has incorrect dimensions"
        assert self.R.shape == (self.nu, self.nu), "R matrix has incorrect dimensions"
        assert self.P.shape == (self.nx, self.nx), "P matrix has incorrect dimensions"
        assert self.Rd.shape == (self.nu, self.nu), "Rd matrix has incorrect dimensions"
        assert self.x_min.shape == (self.nx,), "x_min has incorrect dimensions"
        assert self.x_max.shape == (self.nx,), "x_max has incorrect dimensions"
        assert self.u_min.shape == (self.nu,), "u_min has incorrect dimensions"
        assert self.u_max.shape == (self.nu,), "u_max has incorrect dimensions"
        assert self.ud_min.shape == (self.nu,), "ud_min has incorrect dimensions"
        assert self.ud_max.shape == (self.nu,), "ud_max has incorrect dimensions"
        
def solver_config_kinematic(Q, R, Rd, P, x_min, x_max, u_min, u_max):
    """
    Generate the recommended solver configuration object for a kinematic bicycle model.

    Args:
        Q: float - state cost matrix
        R: float - input cost matrix
        Rd: float - input rate cost matrix
        Qf: float - final state cost matrix
        x_min: np.ndarray - state lower bounds
        x_max: np.ndarray - state upper bounds
        u_min: np.ndarray - input lower bounds
        u_max: np.ndarray - input upper bounds
    """
    DT = 0.1
    N = 10
    nx = 4
    nu = 2
    return solver_config(DT, N, nx, nu, Q, R, Rd, P, x_min, x_max, u_min, u_max)

def solver_config_dynamic(Q, R, Rd, P, x_min, x_max, u_min, u_max):
    """
    Generate the recommended solver configuration object for a dynamic bicycle model.

    Args:
        Q: float - state cost matrix
        R: float - input cost matrix
        Rd: float - input rate cost matrix
        P: float - final state cost matrix
        x_min: np.ndarray - state lower bounds
        x_max: np.ndarray - state upper bounds
        u_min: np.ndarray - input lower bounds
        u_max: np.ndarray - input upper bounds
    """
    DT = 0.025
    N = 40
    nx = 7
    nu = 2
    return solver_config(DT, N, nx, nu, Q, R, Rd, P, x_min, x_max, u_min, u_max)

