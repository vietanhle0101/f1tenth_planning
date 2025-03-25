"""
NMPC waypoint tracker using CasADi. On init, takes in model equation. 
"""
from dataclasses import dataclass, field
import numpy as np
from f1tenth_planning.control.controller import Controller
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.config.solver_config import solver_config
from f1tenth_planning.control.discretizers import rk4_discretization
from f1tenth_planning.utils.utils import nearest_point
from f1tenth_gym.envs.track import Track
import casadi as ca

class Nonlinear_MPC_Solver:
    """
    NMPC Solver, uses CasADi to solve the nonlinear MPC problem using whatever model is passed in.

    Args:
        track (f1tenth_gym_ros:Track): track object, contains the reference raceline
        config (mpc_config, optional): MPC configuration object, contains MPC costs and constraints
        model (Dynamics_Model): dynamics model object, contains the vehicle dynamics
        ipopt_opts (dict, optional): options for the IPOPT solver
    """

    def __init__(self, config: solver_config, model: Dynamics_Model, ipopt_opts: dict) -> None:
        self.config = config
        self.model = model
        self.ipopt_opts = ipopt_opts
        self.discretizer = rk4_discretization
        self.init_problem()

    def init_problem(self):
        
        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym('X', self.config.nx, self.config.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym('U', self.config.nu, self.config.N)

        # coloumn vector for storing initial state and target state
        Params = ca.SX.sym('Params', self.config.nx, self.config.N+1)

        # state weights matrix converted from config Qk
        Q = ca.diagcat(*np.diag(self.config.Q))

        # controls weights matrix
        R = ca.diagcat(*np.diag(self.config.R))

        states = ca.SX.sym('states', self.config.nx, 1)
        controls = ca.SX.sym('controls', self.config.nu, 1)
        
        # System dynamics function
        f = self.model.f_casadi()

        cost_fn = 0  # cost function
        g = X[:, 0] - Params[:, 0]  # x(0) = x0 constraint in the equation

        # loop over all time steps
        for k in range(self.config.N):
            st = X[:, k]
            con = U[:, k]

            # state tracking cost + input cost
            cost_fn = cost_fn \
                + (st - Params[:, k+1]).T @ Q @ (st - Params[:, k+1]) \
                + con.T @ R @ con    
            
            # state dynamics constraint
            st_next = X[:, k+1]
            st_next_RK4 = self.discretizer(f, st, con, self.config.DT)
            g = ca.vertcat(g, st_next - st_next_RK4)

        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )
        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': Params
        }

        # Solver initialization, this is the main solver for the NMPC problem which will be called at each time step
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, self.ipopt_opts)
        
        # Lower and upper bounds for state and control variables
        lbx = ca.vertcat(
            ca.repmat(self.config.x_min, self.config.N + 1, 1),
            ca.repmat(self.config.u_min, self.config.N, 1)
        )
        ubx = ca.vertcat(
            ca.repmat(self.config.x_max, self.config.N + 1, 1),
            ca.repmat(self.config.u_max, self.config.N, 1)
        )

        # lbg is all zeros
        lbg = ca.vertcat(
            ca.DM.zeros((self.config.nx*(self.config.N+1), 1)),
        )
        ubg = ca.vertcat(
            ca.DM.zeros((self.config.nx*(self.config.N+1), 1)),
        )

        # store the arguments for the solver, these are updated at each time step
        self.args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }
        self.U0 = ca.DM.zeros((self.config.nu, self.config.N))

        return
    
    def solve(self, x0, xref):
        self.args['p'] = ca.horzcat(
            x0,    # current state
            xref[:, 1:]  # reference states
        )
        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(ca.repmat(x0, 1, self.config.N+1), self.config.nx*(self.config.N+1), 1),
            ca.reshape(self.U0, self.config.nu*self.config.N, 1)
        )
        sol = self.solver(
            x0=self.args['x0'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            p=self.args['p']
        )
        
        u_sol = ca.reshape(sol['x'][self.config.nx*(self.config.N+1):], self.config.nu, self.config.N)
        x_sol = ca.reshape(sol['x'][:self.config.nx*(self.config.N+1)], self.config.nx, self.config.N+1)

        return x_sol, u_sol