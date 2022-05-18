import jax
import jax.numpy as jnp
import numpy as np
import itertools
import functools

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
  return np.vdot(np.conjugate(psi_vec), op_psi_vec) / (psi_norm **2)

def exact_overlap(v1, v2):
  norm_1 = np.vdot(v1, v1)
  norm_2 = np.vdot(v2, v2)
  return np.abs(np.vdot(v1, v2) / np.sqrt(norm_1 * norm_2))  

def _get_overlap_matrix(vectors):
  """ Given a list of array (`vectors`) or an array whose zeroth-dimension is batch dimension (representing a stacked vector)."""
  num_vecs = len(vectors)
  indices_list = list(itertools.combinations_with_replacement(range(num_vecs), 2))
  overlap_mat = np.zeros((num_vecs, num_vecs))
  for index_pair in indices_list:
    overlap = exact_overlap(vectors[index_pair[0]], vectors[index_pair[1]])
    overlap_mat[index_pair] = overlap
    overlap_mat[index_pair[1], index_pair[0]] = overlap
  return overlap_mat  