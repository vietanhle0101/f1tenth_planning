from f1tenth_gym.envs.track import Track
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.config.dynamics_config import dynamics_config

import numpy as np
import casadi as ca

class Dynamic_Bicycle_Model(Dynamics_Model):
    """
    Dynamic single-track bicycle model for vehicle dynamics. 

    Vehicle State: 
        - [x, y, delta, v, yaw, yaw_rate, slip_angle]
    Control:
        - [delta_v, a]
    
    Reference:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf?ref_type=heads

    Args:
        dynamics_config: dynamics_config - vehicle dynamics configuration
    """
    def __init__(self, track: Track, params: dynamics_config):
        super().__init__(track, params)     
        self.nx = 7
        self.nu = 2   

    def f(self, state: dict, control: np.ndarray, params: dynamics_config = None) -> np.ndarray:
        """
        Compute the state derivative given the current state and control input.

        Args:
            state (np.ndarray): dynamic state as [x, y, delta, v, yaw, yaw_rate, slip_angle]
            control (np.ndarray): control input as (steering_velocity, acceleration)
            params (dynamics_config): vehicle dynamics parameters

        Returns:
            np.ndarray: state derivative
        """
        if params is not None:
            self.params = params

        x, y, delta, v, yaw, yaw_rate, slip_angle = state
        delta_v, a = control

        # Compute the state derivative
        dx = v * np.cos(yaw + slip_angle)
        dy = v * np.sin(yaw + slip_angle)
        ddelta = delta_v
        dv = a

        dyaw = 0
        ddyaw = 0
        dslip_angle = 0
        if np.abs(v) <= 0.1:
            # derivative of yaw "kinemaitcally"
            dyaw = v * np.cos(slip_angle) / self.params.WHEELBASE * np.tan(delta)

            # derivative of slip angle and yaw rate
            dslip_angle = (self.params.LR * delta_v) / (self.params.WHEELBASE * np.cos(delta) ** 2 * (1 + (np.tan(delta) * self.params.LR / self.params.WHEELBASE) ** 2))
            ddyaw = 1 / self.params.WHEELBASE * (a * np.cos(slip_angle) * np.tan(delta) -
                                v * np.sin(slip_angle) * np.tan(delta) * dslip_angle +
                                v * np.cos(slip_angle) * delta_v / np.cos(delta) ** 2)
        else:
            dyaw = yaw_rate

            # Extract params for more readable equations
            mu = self.params.MU
            m = self.params.M
            I = self.params.I
            lr = self.params.LR
            lf = self.params.LF
            C_Sf = self.params.C_SF
            C_Sr = self.params.C_SR
            h = self.params.H
            g = 9.81

            ddyaw = -mu * m / (v * I * (lr + lf)) * (lf ** 2 * C_Sf * (g * lr - a * h) + lr ** 2 * C_Sr * (g * lf + a * h)) * yaw_rate + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + a * h) - lf * C_Sf * (g * lr - a * h)) * slip_angle + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - a * h) * delta

            dslip_angle = (mu / (v ** 2 * (lr + lf)) * (C_Sr * (g * lf + a * h) * lr - C_Sf * (g * lr - a * h) * lf) - 1) * yaw_rate - mu / (v * (lr + lf)) * (C_Sr * (g * lf + a * h) + C_Sf * (g * lr - a * h)) * slip_angle + mu / (v * (lr + lf)) * (C_Sf * (g * lr - a * h)) * delta

        return np.array([dx, dy, ddelta, dv, dyaw, ddyaw, dslip_angle])
    
    def f_casadi(self) -> ca.Function:
        # State symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        delta = ca.SX.sym('delta')
        v = ca.SX.sym('v')
        yaw = ca.SX.sym('yaw')
        yaw_rate = ca.SX.sym('yaw_rate')
        slip_angle = ca.SX.sym('slip_angle')
        states = ca.vertcat(
            x,
            y,
            delta,
            v,
            yaw,
            yaw_rate,
            slip_angle
        )
        # control symbolic variables
        a = ca.SX.sym('a')
        delta_v = ca.SX.sym('delta_v')
        controls = ca.vertcat(
            delta_v,
            a
        )

        # parameters symbolic variables
        mu = ca.SX.sym('mu')
        m = ca.SX.sym('m')
        I = ca.SX.sym('I')
        lr = ca.SX.sym('lr')
        lf = ca.SX.sym('lf')
        C_Sf = ca.SX.sym('C_Sf')
        C_Sr = ca.SX.sym('C_Sr')
        h = ca.SX.sym('h')
        g = ca.SX.sym('g')
        params = ca.vertcat(
            mu,
            m,
            I,
            lr,
            lf,
            C_Sf,
            C_Sr,
            h,
            g
        )

        # right-hand side of the equation
        RHS = self.f_casadi_opti(states, controls, params)

        # maps controls, states and parameters to the right-hand side of the equation
        f = ca.Function('f', [states, controls, params], [RHS])
        return f
        
    def f_casadi_opti(self, state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX:
        # Extract params for more readable equations
        mu = params[0]
        m = params[1]
        I = params[2]
        lr = params[3]
        lf = params[4]
        C_Sf = params[5]
        C_Sr = params[6]
        h = params[7]
        g = params[8]
        
        # Extract state variables from x 
        x = state[0]
        y = state[1]
        delta = state[2]
        v = state[3]
        yaw = state[4]
        yaw_rate = state[5]
        slip_angle = state[6]

        # Extract control variables from u
        delta_v = control[0]
        a = control[1]

        dyaw_slow = v * ca.cos(slip_angle) * ca.tan(delta) / (lr + lf)
        d_beta_slow = (lr * delta_v) / ((lr + lf) * ca.cos(delta) ** 2 * (1 + (ca.tan(delta) ** 2 * lr / (lr + lf)) ** 2))
        dyaw_rate_slow = 1 / (lr + lf) * (a * ca.cos(slip_angle) * ca.tan(delta) -
                            v * ca.sin(slip_angle) * ca.tan(delta) * d_beta_slow  +
                            v * ca.cos(slip_angle) * delta_v / (ca.cos(delta) ** 2))
        
        epsilon = 1e-4
        dyaw_fast = yaw_rate                # dyaw/dt = yaw_rate
        glr = g * lr - a * h
        glf = g * lf + a * h
        # system dynamics
        dyaw_rate_fast = (mu * m/ (I * (lr + lf))) * (
                            lf * C_Sf * (glr) * delta
                            + (lr * C_Sr * (glf) - lf * C_Sf * (glr)) * slip_angle
                            - (lf ** 2 * C_Sf * (glr) + lr ** 2 * C_Sr * (glf)) * (yaw_rate / (v + epsilon))
                        )
        d_beta_fast = (mu / ((v + epsilon) * (lf + lr))) * (
                        C_Sf * (glr) * delta - (C_Sr * (glf) + C_Sf * (glr)) * slip_angle \
                        + (C_Sr * (glf) * lr - C_Sf * (glr) * lf) * (yaw_rate / (v + epsilon)))  - yaw_rate
                                                            
        RHS_LOW_SPEED = ca.vertcat(
                            v * ca.cos(yaw),  # dx/dt = v * cos(yaw + slip_angle)
                            v * ca.sin(yaw),  # dy/dt = v * sin(yaw + slip_angle)
                            delta_v,                 # d(delta)/dt = delta_v
                            a,                       # dv/dt = a
                            dyaw_slow,  # dyaw/dt = yaw_rate
                            dyaw_rate_slow,                  # dyaw_rate/dt = RHS
                            d_beta_slow                   # dbeta/dt = d_beta
                        ) # dx/dt = f(x,u)
    
        RHS_HIGH_SPEED = ca.vertcat(
                            v * ca.cos(yaw + slip_angle),  # dx/dt = v * cos(yaw + slip_angle)
                            v * ca.sin(yaw + slip_angle),  # dy/dt = v * sin(yaw + slip_angle)
                            delta_v,                 # d(delta)/dt = delta_v
                            a,                       # dv/dt = a
                            dyaw_fast,  # dyaw/dt = yaw_rate
                            dyaw_rate_fast,                  # dyaw_rate/dt = RHS
                            d_beta_fast                   # dbeta/dt = d_beta
                        ) # dx/dt = f(x,u)

        RHS = ca.if_else(v >= 0.5, RHS_HIGH_SPEED, RHS_LOW_SPEED)

        return RHS
    
    def parameters_vector_from_config(self, params):
        return np.array([
            params.MU,
            params.M,
            params.I,
            params.LR,
            params.LF,
            params.C_SF,
            params.C_SR,
            params.H,
            9.81
        ])

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters for the dynamic model.
        """
        active_params = [
            self.params.MU,
            self.params.M,
            self.params.I,
            self.params.LR,
            self.params.LF,
            self.params.C_SF,
            self.params.C_SR,
            self.params.H,
            self.params.MU,
        ]
        return len(active_params)
    
    def linearize_around_state(self, state: np.ndarray, control: np.ndarray, params: dynamics_config = None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Linearization not implemented for dynamic model yet.")
        x, y, delta, v, yaw, yaw_rate, slip_angle = state
        delta_v, a = control

        # State (or system) matrix A, 7x7
        A = np.zeros((self.nx, self.nx))

        # dx/dstate
        A[0, 3] = np.cos(yaw + slip_angle)                                             # dx/d(v)
        A[0, 4] = -v * np.sin(yaw + slip_angle)                                        # dx/d(yaw)
        A[0, 6] = -v * np.sin(yaw + slip_angle)                                        # dx/d(slip_angle)

        # dy/dstate
        A[1, 3] = np.sin(yaw + slip_angle)                                             # dy/d(v)
        A[1, 4] = v * np.cos(yaw + slip_angle)                                         # dy/d(yaw)
        A[1, 6] = v * np.cos(yaw + slip_angle)                                         # dy/d(slip_angle)

        if(np.abs(v) <= 0.1):
            # dyaw/dstate
            A[4, 2] = v * np.cos(slip_angle) / self.params.WHEELBASE * (1 / np.cos(delta) ** 2) # dyaw/ddelta
            A[4, 3] = np.cos(slip_angle) / self.params.WHEELBASE * np.tan(delta)                # dyaw/dv
            A[4, 6] = -v * np.sin(slip_angle) / self.params.WHEELBASE * np.tan(delta)            # dyaw/dslip_angle

            # ddyaw/dstate (self.params.LR * delta_v) / (self.params.WHEELBASE * np.cos(delta) ** 2 * (1 + (np.tan(delta) * self.params.LR / self.params.WHEELBASE) ** 2))
            A[5, 2] = (self.params.LR * delta_v) / (self.params.WHEELBASE * np.cos(delta) ** 2 * (1 + (np.tan(delta) * self.params.LR / self.params.WHEELBASE) ** 2)) # ddyaw/ddelta

            pass
        else:
            # Extract params for more readable equations
            mu = self.params.MU
            m = self.params.M
            I = self.params.I
            lr = self.params.LR
            lf = self.params.LF
            C_Sf = self.params.C_SF
            C_Sr = self.params.C_SR
            h = self.params.H
            g = 9.81

            # dyaw/dstate
            A[4, 5] = 1

            # ddyaw/dstate

            pass

        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = 1

        return A, B