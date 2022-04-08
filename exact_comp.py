import jax
import jax.numpy as jnp
import numpy as np
import itertools
# import notebook_fn

def get_vector(num_sites, batch_size, psi, psi_params):
  """Generates a full wavefunction by evaluating `psi` on basis elements."""
  # print(basis_iterator)
  psi_fn = jax.jit(jax.vmap(functools.partial(psi, psi_params)))
  psi_values = []
  basis_iterator = _get_full_basis_iterator(num_sites, batch_size)
  for cs in basis_iterator:
    psi_values.append(jax.device_get(psi_fn(cs)))
  return np.concatenate(psi_values)  

def _get_full_basis_iterator(n_sites, batch_size):
  iterator = itertools.product([-1., 1.], repeat=n_sites)
  return _batch_iterator(iterator, batch_size)

def _batch_iterator(iterator, batch_size=1):
  cs = []
  count = 0
  for c in iterator:
    cs.append(c)
    count += 1
    if count == batch_size:
      cs_out = np.stack(cs)
      cs = []
      count = 0
      yield cs_out
  if cs:
    yield np.stack(cs)

def compute_op_fn(psi_param, psi, op, num_sites, batch_size):
  psi_vec = get_vector(num_sites, batch_size, psi, psi_param)
  psi_norm = np.linalg.norm(psi_vec)
  op_psi = op.get_apply_psi(psi)
  op_psi_vec = get_vector(num_sites, batch_size, op_psi, psi_param)
  op_psi_vec_norm = np.linalg.norm(op_psi_vec)
  return np.vdot(psi_vec, op_psi_vec) / (psi_norm * op_psi_vec_norm)