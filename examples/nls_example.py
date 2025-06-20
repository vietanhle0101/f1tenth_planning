import casadi as ca
import numpy as np

def f_casadi_opti(state: ca.SX, control: ca.SX, params: ca.SX) -> ca.SX:
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
    d_beta_slow = (lr * delta_v) / (
        (lr + lf)
        * ca.cos(delta) ** 2
        * (1 + (ca.tan(delta) ** 2 * lr / (lr + lf)) ** 2)
    )
    dyaw_rate_slow = (
        1
        / (lr + lf)
        * (
            a * ca.cos(slip_angle) * ca.tan(delta)
            - v * ca.sin(slip_angle) * ca.tan(delta) * d_beta_slow
            + v * ca.cos(slip_angle) * delta_v / (ca.cos(delta) ** 2)
        )
    )

    epsilon = 1e-4
    dyaw_fast = yaw_rate  # dyaw/dt = yaw_rate
    glr = g * lr - a * h
    glf = g * lf + a * h
    # system dynamics
    dyaw_rate_fast = (mu * m / (I * (lr + lf))) * (
        lf * C_Sf * (glr) * delta
        + (lr * C_Sr * (glf) - lf * C_Sf * (glr)) * slip_angle
        - (lf**2 * C_Sf * (glr) + lr**2 * C_Sr * (glf)) * (yaw_rate / (v + epsilon))
    )
    d_beta_fast = (mu / ((v + epsilon) * (lf + lr))) * (
        C_Sf * (glr) * delta
        - (C_Sr * (glf) + C_Sf * (glr)) * slip_angle
        + (C_Sr * (glf) * lr - C_Sf * (glr) * lf) * (yaw_rate / (v + epsilon))
    ) - yaw_rate

    RHS_LOW_SPEED = ca.vertcat(
        v * ca.cos(yaw),  # dx/dt
        v * ca.sin(yaw),  # dy/dt
        delta_v,  # d(delta)/dt
        a,  # dv/dt
        dyaw_slow,  # dyaw/dt
        dyaw_rate_slow,  # dyaw_rate/dt
        d_beta_slow,  # dbeta/dt
    )

    RHS_HIGH_SPEED = ca.vertcat(
        v * ca.cos(yaw + slip_angle),  # dx/dt
        v * ca.sin(yaw + slip_angle),  # dy/dt
        delta_v,  # d(delta)/dt
        a,  # dv/dt
        dyaw_fast,  # dyaw/dt
        dyaw_rate_fast,  # dyaw_rate/dt
        d_beta_fast,  # dbeta/dt
    )

    RHS = ca.if_else(v >= 0.5, RHS_HIGH_SPEED, RHS_LOW_SPEED)

    return RHS

def f_casadi() -> ca.Function:
    # State symbolic variables
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    delta = ca.SX.sym("delta")
    v = ca.SX.sym("v")
    yaw = ca.SX.sym("yaw")
    yaw_rate = ca.SX.sym("yaw_rate")
    slip_angle = ca.SX.sym("slip_angle")
    states = ca.vertcat(x, y, delta, v, yaw, yaw_rate, slip_angle)
    # control symbolic variables
    a = ca.SX.sym("a")
    delta_v = ca.SX.sym("delta_v")
    controls = ca.vertcat(delta_v, a)

    # parameters symbolic variables
    mu = ca.SX.sym("mu")
    m = ca.SX.sym("m")
    I = ca.SX.sym("I")
    lr = ca.SX.sym("lr")
    lf = ca.SX.sym("lf")
    C_Sf = ca.SX.sym("C_Sf")
    C_Sr = ca.SX.sym("C_Sr")
    h = ca.SX.sym("h")
    g = ca.SX.sym("g")
    params = ca.vertcat(mu, m, I, lr, lf, C_Sf, C_Sr, h, g)

    # right-hand side of the equation
    RHS = f_casadi_opti(states, controls, params)

    # maps controls, states and parameters to the right-hand side of the equation
    f = ca.Function("f", [states, controls, params], [RHS])
    return f

# RK4
def rk4_step(x, u, p, dt, f):
    k1 = f(x, u, p)
    k2 = f(x + 0.5 * dt * k1, u, p)
    k3 = f(x + 0.5 * dt * k2, u, p)
    k4 = f(x + dt * k3, u, p)
    return x + ((dt * (k1 + 2 * k2 + 2 * k3 + k4)) / 6)

# Instantiate the CasADi dynamics function
f_dynamics = f_casadi()

# dummy data
n_samples = 200 # Number of data points
n_states = 7
n_controls = 2
dt = 0.05 # Time step between data points

true_params_dict = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "m": 3.74,
    "I": 0.04712,
    "g": 9.81,
}
true_params_values = ca.DM([true_params_dict[key] for key in ['mu', 'm', 'I', 'lr', 'lf', 'C_Sf', 'C_Sr', 'h', 'g']])
X_k = np.zeros((n_samples, n_states))
U_k = np.zeros((n_samples, n_controls))
X_k_plus_1 = np.zeros((n_samples, n_states))
X_k[0, :] = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0]) # x, y, delta, v, yaw, yaw_rate, slip_angle

# Simulation
for k in range(n_samples - 1):
    U_k[k, 0] = 0.02 * np.sin(k * dt * 5) # delta_v
    U_k[k, 1] = 0.8 + 0.2 * np.cos(k * dt * 2) # a_x
    X_k_plus_1[k, :] = rk4_step(X_k[k, :], U_k[k, :], true_params_values, dt, f_dynamics).full().flatten()
    X_k_plus_1[k, :] += np.random.randn(n_states) * 0.05
    X_k[k+1, :] = X_k_plus_1[k, :]

# Adjust arrays to match the loop (last X_k and U_k have no corresponding next_X_k_plus_1)
X_k = X_k[:-1, :]
U_k = U_k[:-1, :]
X_k_plus_1 = X_k_plus_1[:-1, :]


# --- Set up CasADi Opti environment for NLS ---
opti = ca.Opti()
mu_est = opti.variable()
m_est = opti.variable()
I_est = opti.variable()
lr_est = opti.variable()
lf_est = opti.variable()
C_Sf_est = opti.variable()
C_Sr_est = opti.variable()
h_est = opti.variable()
g_est = opti.variable()

# Vertically concatenate them to form the 'params' vector for your f_dynamics function
params_est = ca.vertcat(mu_est, m_est, I_est, lr_est, lf_est, C_Sf_est, C_Sr_est, h_est, g_est)

# --- objective function ---
objective = 0
num_points = X_k.shape[0]
for k in range(num_points):
    # Extract current state and control from the data
    current_X_point = X_k[k, :]
    current_U_point = U_k[k, :]
    observed_next_X_point = X_k_plus_1[k, :]

    # Predict the next state using RK4 integration
    predicted_next_X_model = rk4_step(current_X_point, current_U_point, params_est, dt)
    error_vector = predicted_next_X_model - observed_next_X_point
    objective += ca.sumsqr(error_vector)

opti.minimize(objective)

# --- Initial guesses and constraints ---
opti.set_initial(mu_est, 1.0)
opti.set_initial(m_est, 4.0)
opti.set_initial(I_est, 0.05)
opti.set_initial(lr_est, 0.2)
opti.set_initial(lf_est, 0.15)
opti.set_initial(C_Sf_est, 5.0)
opti.set_initial(C_Sr_est, 5.0)
opti.set_initial(h_est, 0.08)
opti.set_initial(g_est, 9.81)

# Parameters must be positive
opti.subject_to(mu_est > 0)
opti.subject_to(m_est > 0)
opti.subject_to(I_est > 0)
opti.subject_to(lr_est > 0)
opti.subject_to(lf_est > 0)
opti.subject_to(C_Sf_est > 0)
opti.subject_to(C_Sr_est > 0)
opti.subject_to(h_est > 0)
opti.subject_to(g_est > 0)
opti.solver('ipopt')

# --- Solve ---
print("\nStarting nonlinear least squares optimization...")
try:
    sol = opti.solve()
    print("Optimization finished successfully.")
    estimated_params = {
        'mu': sol.value(mu_est),
        'm': sol.value(m_est),
        'I': sol.value(I_est),
        'lr': sol.value(lr_est),
        'lf': sol.value(lf_est),
        'C_Sf': sol.value(C_Sf_est),
        'C_Sr': sol.value(C_Sr_est),
        'h': sol.value(h_est),
        'g': sol.value(g_est)
    }

    print("\n--- Estimated Parameters (Nonlinear Least Squares) ---")
    for param, value in estimated_params.items():
        print(f"{param}: {value:.4f}")

    print("\n--- True Parameters (for comparison, from dummy data generation) ---")
    for param, value in true_params_dict.items():
        print(f"{param}: {value:.4f}")

except Exception as e:
    print(f"\nOptimization failed: {e}")
    print("Consider adjusting initial guesses, bounds, or data quality.")
    # You can access the last primal solution even if solver fails, e.g., opti.debug.value(mu_est)