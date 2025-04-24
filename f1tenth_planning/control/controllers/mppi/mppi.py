import jax
import jax.numpy as jnp
from functools import partial
from f1tenth_planning.control.dynamics_model import Dynamics_Model
from f1tenth_planning.control.config.controller_config import mppi_config
from f1tenth_planning.control.discretizers import rk4_discretization
import numpy as np

class MPPI:
  """
  Path-tracking Model Predictive Path Integral (MPPI) controller.
  paper: https://arxiv.org/pdf/1707.02342 | base code: https://github.com/google-research/google-research/tree/master/jax_mpc
  
  Args:
      config (mppi_config): MPPI configuration object, contains MPPI costs and constraints
      model (Dynamics_Model): dynamics model object, used to compute the state derivative
  """

  def __init__(self, config: mppi_config, model: Dynamics_Model):
    self.config = config
    self.model = model
    self.discretizer = rk4_discretization
    self.control_params = self._init_control()  # [N, nu]

  def _init_control(self):
    """
    Initialize the control parameters for MPPI.

    Returns:
        tuple: (a_opt, a_cov) where a_opt is the optimal control input and a_cov is the covariance matrix.
    """

    a_opt = jnp.zeros((self.config.N, self.config.nu))  # [N, nu]
    # a_cov: [N, nu, nu]
    if self.config.adaptive_covariance:
      # note: should probably store factorized cov,
      # e.g. cholesky, for faster sampling
      a_cov = (self.config.u_std**2)*jnp.tile(jnp.eye(jnp.array(self.config.nu)), (self.config.N, 1, 1))
    else:
      a_cov = None
    return (a_opt, a_cov)

                                               
  @partial(jax.jit, static_argnums=(0))
  def iteration_step(self, input_, env_state, ref_traj):
    a_opt, a_cov, rng = input_
    rng_da, rng = jax.random.split(rng)
    if self.config.adaptive_covariance:
      # Truncated normal to ensure control inputs are within bounds
      sqrt_cov = jnp.linalg.cholesky(a_cov)
      # Must ensure that a + da is within bounds
      lower = self.config.u_min - a_opt
      upper = self.config.u_max - a_opt
      da = sqrt_cov @ jax.random.truncated_normal(
          rng_da,
          lower=lower,
          upper=upper,
          shape=(self.config.n_samples, self.config.N, self.nu)
      ) # [n_samples, N, nu]
    else:
      # Truncated normal to ensure control inputs are within bounds
      sqrt_cov = jnp.linalg.cholesky(a_cov)
      # Must ensure that a + da is within bounds
      lower = self.config.u_min - a_opt
      upper = self.config.u_max - a_opt
      da = sqrt_cov @ jax.random.truncated_normal(
          rng_da,
          lower=lower,
          upper=upper,
          shape=(self.config.n_samples, self.config.N, self.nu)
      ) # [n_samples, N, nu]


    # a: [n_samples, N, nu]
    a = jnp.expand_dims(a_opt, axis=0) + da  # [n_samples, N, nu]
    s, r = jax.vmap(self._rollout, in_axes=(0, None, 0))(
        a, env_state, ref_traj
    )  # [n_samples, N]
    R = jax.vmap(self._returns)(r)  # [n_samples, N], pylint: disable=invalid-name
    w = jax.vmap(self._weights, 1, 1)(R)  # [n_samples, N]
    da_opt = jax.vmap(jnp.average, (1, None, 1))(da, 0, w)  # [N, nu]
    a_opt = a_opt + da_opt # [N, nu]
    if self.config.adaptive_covariance:
      a_cov = jax.vmap(jax.vmap(jnp.outer))(
          da, da
      )  # [n_samples, N, nu, nu]
      a_cov = jax.vmap(jnp.average, (1, None, 1))(
          a_cov, 0, w
      )  # a_cov: [N, nu, nu]
      # prevent loss of rank when one sample is heavily weighted
      a_cov = a_cov + jnp.eye(self.nu)*0.00001
    return (a_opt, a_cov, rng), (a, s, r)
  
  def _step(self, x, u):
    """
    Single-step state prediction function.
    """
    return self.discretizer(self.model.f_jax, x, u, self.model.params)
  
  def _reward(self, x, u, x_ref):
    """
    Single-step reward calculated as the negative of the trajectory tracking error (x^T Q x + u^T R u).
    """
    return - (jnp.dot((x - x_ref).T, jnp.dot(self.config.Q, (x - x_ref))) + jnp.dot(u.T, jnp.dot(self.config.R, u)))

  def update(self, x0, ref_traj, control_params, rng):
    """
    Run the MPPI algorithm for a given initial state and reference trajectory.

    Args:
        x0 (np.ndarray): initial state of shape (nx,)
        ref_traj (np.ndarray): reference trajectory of shape (N+1, nx)
        control_params (tuple): current MPPI control parameters (a_opt, a_cov)
        rng (jax.random.PRNGKey): random key for sampling

    Returns:
        tuple: updated MPPI state and sampled trajectories
    """
    nu = jnp.prod(self.config.nu)  # np.int32
    self.nu = jnp.prod(jnp.array(self.config.nu))

    a_opt, a_cov = control_params
    a_opt = jnp.concatenate([a_opt[1:, :],
                             jnp.expand_dims(jnp.zeros((nu,)),
                                             axis=0)])  # [N, nu]
    if self.config.adaptive_covariance:
      a_cov = jnp.concatenate([a_cov[1:, :],
                               jnp.expand_dims((self.config.u_std**2)*jnp.eye(nu),
                                               axis=0)])
    if not self.config.scan:
      for _ in range(self.config.n_iterations):
        (a_opt, a_cov, rng), (a_sampled, s_sampled, r_sampled) = self.iteration_step((a_opt, a_cov, rng), x0, ref_traj)
    else:
      (a_opt, a_cov, rng), _ = jax.lax.scan(
          lambda input_, _: self.iteration_step(input_, x0, ref_traj), 
          (a_opt, a_cov, rng), None, length=self.config.n_iterations
      )
    return (a_opt, a_cov), (a_sampled, s_sampled, r_sampled)

  @partial(jax.jit, static_argnums=(0))
  def _returns(self, r):
    # r: [N]
    return jnp.dot(jnp.triu(jnp.ones((self.config.N, self.config.N))),
                   r)  # R: [N]

  @partial(jax.jit, static_argnums=(0))
  def _weights(self, R):  # pylint: disable=invalid-name
    # R: [n_samples]
    # R_stdzd = (R - jnp.min(R)) / ((jnp.max(R) - jnp.min(R)) + self.damping)
    # R_stdzd = R - jnp.max(R) # [n_samples] np.float32
    R_stdzd = (R - jnp.max(R)) / ((jnp.max(R) - jnp.min(R)) + self.config.damping)  # pylint: disable=invalid-name
    w = jnp.exp(R_stdzd / self.config.temperature)  # [n_samples] np.float32
    w = w/jnp.sum(w)  # [n_samples] np.float32
    return w

  @partial(jax.jit, static_argnums=(0))
  def _rollout(self, u, x0, xref):
    """
    Rollout the trajectory given the control inputs and initial state.

    Args:
        u (np.ndarray): control inputs of shape (N, nu)
        x0 (np.ndarray): initial state of shape (nx,)
        xref (np.ndarray): reference trajectory of shape (N+1, nx)
    Returns:
        np.ndarray: state trajectory of shape (N+1, nx)
        np.ndarray: reward trajectory of shape (N+1,)
    """

    def rollout_step(x, u, xref):
      u = jnp.reshape(u, jnp.array(self.config.nu))
      x = self._step(x, u)
      r = self._reward(x, u, xref)
      return x, (x, r)
    if not self.config.scan:
      # python equivalent of lax.scan
      scan_output = []
      for t in range(self.config.N):
        x0, output = rollout_step(x0, u[t, :], xref[t, :])
        scan_output.append(output)
      s, r = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *scan_output)
    else:
      _, (s, r) = jax.lax.scan(rollout_step, x0, u, xref)

    return (s, r)
  
  def solve(self, x0, ref_traj)->tuple[jnp.ndarray, jnp.ndarray]:  
    """
    Solve the MPPI problem for the given initial state and reference trajectory.

    Args:
        x0 (np.ndarray): initial state of shape (nx,)
        xref (np.ndarray): reference trajectory of shape (nx, N+1)

    Returns:
        np.ndarray: optimal control input of shape (nu, N)
        np.ndarray: optimal state trajectory of shape (nx, N+1)
    """
    rng = jax.random.PRNGKey(0)
    self.control_params, self.samples = self.update(x0, ref_traj, self.control_params, rng)
  
    # Get the solved for control and state trajectory
    self.uk = self.control_params[0] # [N, nu]
    self.xk = self._rollout(self.uk, x0) # [N, nu]

    return self.xk, self.uk
