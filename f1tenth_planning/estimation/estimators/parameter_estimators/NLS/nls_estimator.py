from __future__ import annotations
from abc import abstractmethod, ABC

import numpy as np
import casadi as ca

from f1tenth_planning.control.config.dynamics_config import dynamics_config
from f1tenth_planning.estimation.estimators.parameter_estimators.paramter_estimator import ParameterEstimator
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.discretizers import rk4_discretization


class NLSParameterEstimator(ParameterEstimator):
    def __init__(self,
                 initial_params: dynamics_config,
                 model: Dynamics_Model,
                 estimation_history_length: int,
                 dt: float = 0.1):
        super().__init__(initial_params)
        self.model = model
        self._N = estimation_history_length
        self._dt = dt
        # histories
        self._state_history   = []
        self._control_history = []
        self._next_history    = []
        # build (and store) the Opti problem one time:
        self._opti  = self.__initialize_nls_problem(initial_params,
                                                    model.nx,
                                                    model.nu,
                                                    estimation_history_length)

    def __initialize_nls_problem(self,
                                 initial_params: dynamics_config,
                                 n_x: int,
                                 n_u: int,
                                 N: int):
        """
        Build a single Opti() problem that has:
          - decision vars = your parameters
          - parameters   = the last N rows of (X_k, U_k, X_{k+1})
          - objective    = sum of squared RK4 errors
        """
        opti = ca.Opti()

        # decision vars for each param, enforce positivity
        self._param_vars = []
        for field, val in vars(initial_params).items():
            v = opti.variable()
            opti.subject_to(v > 0)
            opti.set_initial(v, val)
            self._param_vars.append(v)
        self._param_vars = ca.vertcat(*self._param_vars)

        # placeholders for data
        self._Xk       = opti.parameter(self._N, n_x)
        self._Uk       = opti.parameter(self._N, n_u)
        self._Xkp1     = opti.parameter(self._N, n_x)

        f_casadi = self.model.f_casadi()
        obj = 0
        for i in range(self._N):
            x_k  = self._Xk[i, :]
            u_k  = self._Uk[i, :]
            x_k_plus_1 = self._Xkp1[i, :]
            x_pred = rk4_discretization(x_k, u_k, self._param_vars, self._dt, f_casadi)
            err = x_pred - x_k_plus_1
            obj += err
        opti.minimize(obj)

        opti.solver('ipopt')
        return opti

    def estimate(self,
                 state: dict,
                 control: np.ndarray) -> dynamics_config:
        """
        Append one step of (x_{k-1}, u_{k-1}, x_k) to the history
        and re-solve for the best parameters.
        The user only passes the current state and control.
        """
        x = state["pose_x"]
        y = state["pose_y"]
        v = state["linear_vel_x"]
        yaw = state["pose_theta"]
        x_curr = np.array([x, y, state["delta"], v, yaw, state["ang_vel_z"], state["beta"]]) # Currently only support dynamic-bycicle model

        # on first call, just store and return initial params
        if not hasattr(self, "_prev_state") or self._prev_state is None:
            self._prev_state = x_curr
            self._prev_control = control
            return self.params

        # build one data tuple: (previous state, previous control, current state)
        x_k  = self._prev_state
        u_k  = self._prev_control
        x_k1 = x_curr

        # store for next iteration
        self._prev_state   = x_curr
        self._prev_control = control

        # append to histories
        self._state_history.append(x_k)
        self._control_history.append(u_k)
        self._next_history.append(x_k1)

        # keep at most N entries
        if len(self._state_history) > self._N:
            self._state_history.pop(0)
            self._control_history.pop(0)
            self._next_history.pop(0)

        M = len(self._state_history)
        if M < self._N:
            return self.params  # not enough data to estimate

        Xk   = np.vstack(self._state_history)
        Uk   = np.vstack(self._control_history)
        Xkp1 = np.vstack(self._next_history)

        self._opti.set_value(self._Xk,   Xk)
        self._opti.set_value(self._Uk,   Uk)
        self._opti.set_value(self._Xkp1, Xkp1)

        # solve the NLS problem
        sol = self._opti.solve()

        # update self.params from solution
        p_vals = sol.value(self._param_vars).full().flatten()
        for i, name in enumerate(vars(self.params).keys()):
            setattr(self.params, name, float(p_vals[i]))

        return self.params
