import math
import cvxpy
import numpy as np
from f1tenth_planning.control.config.solver_config import solver_config
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.discretizers import euler_discretization, system_matrix_discretization
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix
from dataclasses import dataclass, field

class LTV_MPC_Solver:
    """
    Formulates and solves a Linear Time-Varying Model Predictive Control (LTV-MPC) problem for a time-varying system tracking a reference trajectory.
    
    """
    def __init__(self, config: solver_config, model: Dynamics_Model):
        self.config = config
        self.model = model
        self.init_problem()
        self.discretizer = euler_discretization
        self.dynamics_discretizer = system_matrix_discretization

    def init_problem(self):
        """
        Initialize the MPC problem using the solver configuration parameters.
        """
        self.xk = cvxpy.Variable((self.config.nx, self.config.N + 1), name="x[k]")
        self.uk = cvxpy.Variable((self.config.nu, self.config.N), name="u[k]")
        
        self.x0 = cvxpy.Parameter(self.config.nx, name="x[0]")
        self.ref_traj = cvxpy.Parameter((self.config.nx, self.config.N + 1), name="x_ref[k]")

        self.Qk = cvxpy.Parameter((self.config.nx, self.config.nx), name="Q[k]")
        self.Rk = cvxpy.Parameter((self.config.nu, self.config.nu), name="R[k]")
        self.Qf = cvxpy.Parameter((self.config.nx, self.config.nx), name="Qf")

        # Initialize variables
        self.xk.value = np.zeros((self.config.nx,))

        # Initialize reference trajectory parameter
        self.ref_traj.value = np.zeros((self.config.nx, self.config.N + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.R] * self.config.N))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rd] * (self.config.N - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Q] * (self.config.N)
        Q_block.append(self.config.Qf)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj), Q_block)

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        path_predict = np.zeros((self.config.nx, self.config.N + 1))
        input_predict = np.zeros((self.config.nu, self.config.N))
        A_block, B_block, C_block = self.linearize_dynamics_trajectory(path_predict, input_predict)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz, name="A_nnz[k]")
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz, name="Bnnz[k]")
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape, name="C[k]")
        self.Ck_.value = C_block

        # Add dynamics constraints to the optimization problem
        constraints += [
            cvxpy.vec(self.xk[:, 1:])
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1])
            + self.Bk_ @ cvxpy.vec(self.uk)
            + (self.Ck_)
        ]

        # Constraints 2: State and input constraints
        constraints += [self.xk[:, 0] == self.x0k]
        constraints += [self.xk <= self.config.x_max]
        constraints += [self.xk >= self.config.x_min]
        constraints += [self.uk <= self.config.u_max]
        constraints += [self.uk >= self.config.u_min]

        # Constraints 3: Input rate constraints
        constraints += [cvxpy.diff(self.uk, axis=1) <= self.config.ud_max]
        constraints += [cvxpy.diff(self.uk, axis=1) >= self.config.ud_min]

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        
    def solve(self, x0, xref, uref=None, Q=None, Qf=None, R=None)-> tuple[np.ndarray, np.ndarray]:  
        """
        Solve the LTV-MPC problem for the given initial state and reference trajectory.

        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            xref (np.ndarray): reference trajectory of shape (nx, N+1)
            uref (np.ndarray): reference control input of shape (nu, N)
            Q (np.ndarray): state cost matrix
            R (np.ndarray): input cost matrix

        Returns:
            np.ndarray: optimal control input of shape (nu, N)
            np.ndarray: optimal state trajectory of shape (nx, N+1)
        """
        # Set the reference trajectory
        self.ref_traj.value = xref

        # Set the initial state
        self.x0.value = x0

        # Set the cost matrices
        if Q is not None:
            self.Qk.value = Q
            self.config.Q = Q
        if Qf is not None:
            self.Qf.value = Qf
            self.config.Qf = Qf
        if R is not None:
            self.Rk.value = R
            self.config.R = R
        
        # Linearize the dynamics model along the reference trajectory
        A_block, B_block, C_block = self.linearize_dynamics_trajectory(xref, uref)
        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)
        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        # Shifted control and state variables for warm start and fallback in case of optimization failure
        last_u = self.uk.value
        last_x = self.xk.value
        shifted_u = np.hstack((last_u[:, 1:], last_u[:, -1].reshape(-1, 1)))
        pred_x = self.discretizer(self.model.f, last_x[:, -1], shifted_u[:, -1], self.config.DT)
        shifted_x = np.hstack((last_x[:, 1:], pred_x.reshape(-1, 1)))

        # Warm start with shifted control and state variables
        self.xk.value = shifted_x
        self.uk.value = shifted_u

        # Solve the optimization problem
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        
        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            return self.uk.value, self.xk.value
        else:
            print("Optimization problem failed! Returning the last control input shifted by one timestep.")
            return shifted_u, shifted_x


    def predict_state(self, x0, u_traj):
        """
        Predict the system state for the next N steps using the model and control inputs.

        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            u_traj (np.ndarray): control input of shape (nu, N)
        """
        x = x0
        traj_predict = np.zeros((self.config.nx, self.config.N + 1))
        traj_predict[:, 0] = x0
        for i in range(self.config.N):
            x = self.discretizer(self.model.f, x, u_traj[:, i], self.config.DT)
            traj_predict[:, i + 1] = x
        return traj_predict
    
    def get_system_matrices(self, x, u):
        """
        Get the discretized, linearized system matrices for the current state and control input.

        Args:
            x (np.ndarray): state of shape (nx,)
            u (np.ndarray): control input of shape (nu,)

        Returns:
            Ad (np.ndarray): discretized state matrix
            Bd (np.ndarray): discretized input matrix
            Cd (np.ndarray): linearization residual
        """
        A, B = self.model.linearize_around_state(x, u)
        Ad, Bd = self.dynamics_discretizer(A, B, self.config.DT)
        Cd = self.model.f(x, u) - Ad @ x - Bd @ u
        return Ad, Bd, Cd
    
    def linearize_dynamics_trajectory(self, x_traj, u_traj):
        """
        Linearize the vehicle dynamics model along the trajectory.

        Args:
            x0 (np.ndarray): initial state of shape (nx,)
            u_traj (np.ndarray): control input of shape (nu, N)

        Returns:
            Ad_traj (list(np.ndarray)): list of discretized state matrices
            Bd_traj (list(np.ndarray)): list of discretized input matrices
            Cd_traj (list(np.ndarray)): list of linearization residuals
        """
        Ad_traj, Bd_traj, Cd_traj = [], [], []
        for i in range(self.config.N):
            Ad, Bd, Cd = self.get_system_matrices(x_traj[:, i], u_traj[:, i])
            Ad_traj.append(Ad)
            Bd_traj.append(Bd)
            Cd_traj.append(Cd)
        return Ad_traj, Bd_traj, Cd_traj
    