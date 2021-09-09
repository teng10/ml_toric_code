#@title Compute overlap exactly
import jax
import jax.numpy as jnp
import itertools
import sample_utils

def compute_overlap_exact(w_psi, w_phi, psi, phi, num_spins, batch_size=100):
  def overlap_fn(psis, phis):
    return jnp.mean(jnp.conjugate(psis) * phis)
  
  def norm_fn(psis):
    return jnp.mean(jnp.abs(psis)**2)

  # Define vectorized psis
  psi_vectorized = jax.vmap(psi, in_axes=(None, 0))

  phi_vectorized = jax.vmap(phi, in_axes=(None, 0))
  
  # Define batched configs
  iterator = itertools.product([1., -1.], repeat=num_spins)
  configs_all = sample_utils._batch_iterator(iterator, batch_size)
  new_overlaps = []
  new_norm_psis = []
  new_norm_phis = [] 
  batch_sizes = []


  for idx, configs in enumerate(configs_all):
    # Track batch sizes
    config_batch_size = configs.shape[0]
    batch_sizes.append(config_batch_size)

    # Compute psis, phis
    psis = psi_vectorized(w_psi, configs)
    phis = phi_vectorized(w_phi, configs)  

    # Compute overlap and norm
    overlap = overlap_fn(psis, phis)
    norm_psi = norm_fn(psis)
    norm_phi = norm_fn(phis)

    new_overlaps.append(overlap)
    new_norm_psis.append(norm_psi)
    new_norm_phis.append(norm_phi)

  fraction_list = jnp.array(batch_sizes) / batch_size
  overlap = jnp.mean(fraction_list * jnp.array(new_overlaps))
  norm_psi = jnp.mean(fraction_list * jnp.array(new_norm_psis))
  norm_phi = jnp.mean(fraction_list * jnp.array(new_norm_phis))

  return jnp.sqrt(jnp.abs(overlap)**2 / (norm_psi * norm_phi))