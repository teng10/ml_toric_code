import jax
import jax.numpy as jnp
import numpy as np
import notebook_fn

def compute_op_fn(psi_param, psi, op, num_sites, batch_size):
  psi_vec = notebook_fn.get_vector(num_sites, batch_size, psi, psi_param)
  psi_norm = np.linalg.norm(psi_vec)
  op_psi = op.get_apply_psi(psi)
  op_psi_vec = notebook_fn.get_vector(num_sites, batch_size, op_psi, psi_param)
  op_psi_vec_norm = np.linalg.norm(op_psi_vec)
  return np.vdot(psi_vec, op_psi_vec) / (psi_norm * op_psi_vec_norm)