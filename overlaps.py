#@title Compute overlap exactly
import jax
import jax.numpy as jnp
import numpy as np
import itertools
import sample_utils
import utils
import tc_utils
import functools
import mcmc

def compute_overlap_exact(w_psi, w_phi, psi, phi, spin_shape, batch_size=100):
  def overlap_fn(psis, phis):
    return np.mean(np.conjugate(psis) * phis)
  
  def norm_fn(psis):
    return np.mean(np.abs(psis)**2)

  # Define vectorized psis
  psi_vectorized = jax.vmap(psi, in_axes=(None, 0))
  psi_vectorized = jax.jit(psi_vectorized)

  phi_vectorized = jax.vmap(phi, in_axes=(None, 0))
  phi_vectorized = jax.jit(phi_vectorized)
  
  # Define batched configs
  (shape_x, shape_y) = spin_shape
  num_spins = shape_x * shape_y  
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
    psis = jax.device_get(psi_vectorized(w_psi, configs))
    phis = jax.device_get(phi_vectorized(w_phi, configs))  

    # Compute overlap and norm
    overlap = overlap_fn(psis, phis)
    norm_psi = norm_fn(psis)
    norm_phi = norm_fn(phis)

    new_overlaps.append(overlap)
    new_norm_psis.append(norm_psi)
    new_norm_phis.append(norm_phi)

  fraction_list = np.array(batch_sizes) / batch_size
  overlap = np.mean(fraction_list * np.array(new_overlaps))
  norm_psi = np.mean(fraction_list * np.array(new_norm_psis))
  norm_phi = np.mean(fraction_list * np.array(new_norm_phis))

  return np.sqrt(np.abs(overlap)**2 / (norm_psi * norm_phi)), 0, 0      # overlap, std, num_accepts


def compute_overlap_mcmc( 
                        w_psi, 
                        w_phi, 
                        psi_apply, 
                        phi_apply, 
                        spin_shape,   
                        propose_move_fn,                    
                        num_samples,
                        key,                           
                        len_chain, 
                        len_chain_burn, 
                        ):
  """Return |<psi(w_psi)| phi(w_phi)>|, std, (num_accepts) using mcmc sampling estimation."""

  def _psi_phi_ratio_fn(c, psi_w, w_phi, phi_apply):
    """Utility function for computing ratio phi(c)/psi(c)"""
    phi_w = phi_apply(w_phi, c)
    return phi_w / psi_w
  num_spins = spin_shape[0] * spin_shape[1]  
  key_init, key_mcmc = jax.random.split(key, 2)     # Split keys for initializing samples and mcmc process
  cs_init = sample_utils.init_samples(key_init, num_spins, num_samples)     #Create intial chains
  

  # Define update chain function
  update_chain_fn_psi = functools.partial(mcmc.update_chain, 
                                      psi=psi_apply, 
                                      propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample)
  
  update_chain_fn_phi = functools.partial(mcmc.update_chain, 
                                      psi=phi_apply, 
                                      propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample)
  # Vectorize update_chain_fn
  update_chain_fn_psi_vec = jax.vmap(update_chain_fn_psi, in_axes=(0, 0, 0, None, None))
  update_chain_fn_phi_vec = jax.vmap(update_chain_fn_phi, in_axes=(0, 0, 0, None, None))
  # Split key for first burn and evaluation
  key_burn, key_ev = jax.random.split(key_mcmc, 2)
  # Split first burn rngs to psi and phi
  rngs_burn_psi, rngs_burn_phi = utils.split_key(key_burn, np.array([2, num_samples, 2]))
  # vmap psi_apply and phi_apply
  psi_apply_vec = jax.vmap(psi_apply, in_axes=(None, 0))
  phi_apply_vec = jax.vmap(phi_apply, in_axes=(None, 0))
  # Get initial psis and phis
  psis_init = psi_apply_vec(w_psi, cs_init)
  phis_init = phi_apply_vec(w_phi, cs_init)
  # First burns
  cs_burn_psi, psis_burn, _ = update_chain_fn_psi_vec(rngs_burn_psi, cs_init, psis_init, w_psi, len_chain_burn)
  cs_burn_phi, phis_burn, _ = update_chain_fn_phi_vec(rngs_burn_phi, cs_init, phis_init, w_phi, len_chain_burn)
  # Split evaluation rngs
  rngs_ev_psi, rngs_ev_phi = utils.split_key(key_ev, np.array([2, num_samples, 2]))
  # Get mcmc samples for evaluations 
  cs_ev_psi, psis_ev, num_accepts_psi = update_chain_fn_psi_vec(rngs_ev_psi, cs_burn_psi, psis_burn, w_psi, len_chain)
  cs_ev_phi, phis_ev, num_accepts_phi = update_chain_fn_phi_vec(rngs_ev_phi, cs_burn_phi, phis_burn, w_phi, len_chain)  
  # Estimate ratio for <phi/psi>_psi
  phi_psi_ratio = functools.partial(_psi_phi_ratio_fn, w_phi=w_phi, phi_apply=phi_apply)
  phi_psi_ratio_vec = jax.vmap(phi_psi_ratio, in_axes=(0,0))
  phi_psi_ratio_array = phi_psi_ratio_vec(cs_ev_psi, psis_ev)
  # Estimate ratio for <psi/phi>_phi
  psi_phi_ratio = functools.partial(_psi_phi_ratio_fn, w_phi=w_psi, phi_apply=psi_apply)
  psi_phi_ratio_vec = jax.vmap(psi_phi_ratio, in_axes=(0,0))
  psi_phi_ratio_array = psi_phi_ratio_vec(cs_ev_phi, phis_ev)  
  # Compute mean from sampled configs
  psi_ratio_mean = jnp.mean(phi_psi_ratio_array, axis=0)
  psi_ratio_var = jnp.std(phi_psi_ratio_array, axis=0)
  # psi_ratio_mod_sq_mean = jnp.mean(psi_ratio_mod_sq_array, axis=0)
  phi_ratio_mean = jnp.mean(psi_phi_ratio_array, axis=0)
  phi_ratio_var = jnp.std(psi_phi_ratio_array, axis=0)

  # return psi_ratio_mean * jnp.sqrt(psi_ratio_mod_sq_mean)
  std_all = psi_ratio_var * phi_ratio_var + psi_ratio_var * phi_ratio_mean**2 + phi_ratio_var * psi_ratio_mean**2   
  
  return jnp.sqrt(psi_ratio_mean * phi_ratio_mean), std_all / np.sqrt(num_samples), (num_accepts_psi, num_accepts_phi)

# Estimate overlap old
# def compute_overlap_mcmc( 
#                         w_psi, 
#                         w_phi, 
#                         psi_apply, 
#                         phi_apply, 
#                         spin_shape,   
#                         propose_move_fn,                    
#                         num_samples,
#                         key,                           
#                         len_chain, 
#                         len_chain_burn, 
#                         spin_flip_p,
#                         ):
#   """Return |<psi(w_psi)| phi(w_phi)>|, std, (num_accepts) using mcmc sampling estimation."""

#   def _psi_phi_ratio_fn(c, psi_w, w_phi, phi_apply):
#     """Utility function for computing ratio phi(c)/psi(c)"""
#     phi_w = phi_apply(w_phi, c)
#     return phi_w / psi_w
#   num_spins = spin_shape[0] * spin_shape[1]  
#   key_init, key_mcmc = jax.random.split(key, 2)     # Split keys for initializing samples and mcmc process
#   cs_init = sample_utils.init_samples(key_init, num_spins, num_samples)     #Create intial chains
  

#   # Define update chain function
#   update_chain_fn_psi = functools.partial(mcmc.update_chain, 
#                                       psi=psi_apply, 
#                                       propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample, 
#                                       p=spin_flip_p)
  
#   update_chain_fn_phi = functools.partial(mcmc.update_chain, 
#                                       psi=phi_apply, 
#                                       propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample, 
#                                       p=spin_flip_p)
#   # Vectorize update_chain_fn
#   update_chain_fn_psi_vec = jax.vmap(update_chain_fn_psi, in_axes=(0, 0, 0, None, None))
#   update_chain_fn_phi_vec = jax.vmap(update_chain_fn_phi, in_axes=(0, 0, 0, None, None))
#   # Split key for first burn and evaluation
#   key_burn, key_ev = jax.random.split(key_mcmc, 2)
#   # Split first burn rngs to psi and phi
#   rngs_burn_psi, rngs_burn_phi = utils.split_key(key_burn, np.array([2, num_samples, 2]))
#   # vmap psi_apply and phi_apply
#   psi_apply_vec = jax.vmap(psi_apply, in_axes=(None, 0))
#   phi_apply_vec = jax.vmap(phi_apply, in_axes=(None, 0))
#   # Get initial psis and phis
#   psis_init = psi_apply_vec(w_psi, cs_init)
#   phis_init = phi_apply_vec(w_phi, cs_init)
#   # First burns
#   cs_burn_psi, psis_burn, _ = update_chain_fn_psi_vec(rngs_burn_psi, cs_init, psis_init, w_psi, len_chain_burn)
#   cs_burn_phi, phis_burn, _ = update_chain_fn_phi_vec(rngs_burn_phi, cs_init, phis_init, w_phi, len_chain_burn)
#   # Split evaluation rngs
#   rngs_ev_psi, rngs_ev_phi = utils.split_key(key_ev, np.array([2, num_samples, 2]))
#   # Get mcmc samples for evaluations 
#   cs_ev_psi, psis_ev, num_accepts_psi = update_chain_fn_psi_vec(rngs_ev_psi, cs_burn_psi, psis_burn, w_psi, len_chain)
#   cs_ev_phi, phis_ev, num_accepts_phi = update_chain_fn_phi_vec(rngs_ev_phi, cs_burn_phi, phis_burn, w_phi, len_chain)  
#   # Estimate ratio for <phi/psi>_psi
#   phi_psi_ratio = functools.partial(_psi_phi_ratio_fn, w_phi=w_phi, phi_apply=phi_apply)
#   phi_psi_ratio_vec = jax.vmap(phi_psi_ratio, in_axes=(0,0))
#   phi_psi_ratio_array = phi_psi_ratio_vec(cs_ev_psi, psis_ev)
#   # Estimate ratio for <psi/phi>_phi
#   psi_phi_ratio = functools.partial(_psi_phi_ratio_fn, w_phi=w_psi, phi_apply=psi_apply)
#   psi_phi_ratio_vec = jax.vmap(psi_phi_ratio, in_axes=(0,0))
#   psi_phi_ratio_array = psi_phi_ratio_vec(cs_ev_phi, phis_ev)  
#   # Compute mean from sampled configs
#   psi_ratio_mean = jnp.mean(phi_psi_ratio_array, axis=0)
#   psi_ratio_var = jnp.std(phi_psi_ratio_array, axis=0)
#   # psi_ratio_mod_sq_mean = jnp.mean(psi_ratio_mod_sq_array, axis=0)
#   phi_ratio_mean = jnp.mean(psi_phi_ratio_array, axis=0)
#   phi_ratio_var = jnp.std(psi_phi_ratio_array, axis=0)

#   # return psi_ratio_mean * jnp.sqrt(psi_ratio_mod_sq_mean)
#   std_all = psi_ratio_var * phi_ratio_var + psi_ratio_var * phi_ratio_mean**2 + phi_ratio_var * psi_ratio_mean**2   
  
#   return jnp.sqrt(psi_ratio_mean * phi_ratio_mean), std_all / np.sqrt(num_samples), (num_accepts_psi, num_accepts_phi)



# #title MCMC fidelity including first burns
# def compute_fidelity(key, configs_psi, configs_phi, w_psi, w_phi, psi, phi, 
#                          update_chain_fn_psi, update_chain_fn_phi, 
#                          update_chain_fn_psi_burn, update_chain_fn_phi_burn):
#   """
#   Compute fidelity <psi_w1 | phi_w2>
#   """
#   # Set up keys for sampling psi and phi
#   key_psi, key_phi = jax.random.split(key, 2)

#   # Split key for first burn and MCMC
#   key_psi_new, key_psi_burn = jax.random.split(key_psi, 2)
#   key_phi_new, key_phi_burn = jax.random.split(key_phi, 2)

#   # Vectorize first burn update chain function
#   vectorized_update_chain_psi_burn = jax.vmap(update_chain_fn_psi_burn, in_axes=(0, 0, 0, None))
#   vectorized_update_chain_phi_burn = jax.vmap(update_chain_fn_phi_burn, in_axes=(0, 0, 0, None))

#   # Compute psi/phi of configs
#   psi_vectorized = jax.vmap(psi, in_axes=(None, 0))
#   psis = psi_vectorized(w_psi, configs_psi)

#   phi_vectorized = jax.vmap(phi, in_axes=(None, 0))
#   phis = phi_vectorized(w_phi, configs_phi)

#   # First burns
#   num_batch_psi = configs_psi.shape[0]      #batch size
#   num_batch_phi = configs_phi.shape[0]      #batch size
#   rngs_psi_burn = split_key(key_psi_new, jnp.array([num_batch_psi, 2]))
#   rngs_phi_burn = split_key(key_phi_new, jnp.array([num_batch_phi, 2]))

#   psi_configs_burn, psis_burn, _ = vectorized_update_chain_psi_burn(rngs_psi_burn, configs_psi, psis, w_psi) 
#   phi_configs_burn, phis_burn, _ = vectorized_update_chain_phi_burn(rngs_phi_burn, configs_phi, phis, w_phi) 

#   rngs_psi = split_key(key_psi_new, jnp.array([num_batch_psi, 2]))
#   rngs_phi = split_key(key_phi_new, jnp.array([num_batch_phi, 2]))

#   vectorized_update_chain_psi = jax.vmap(update_chain_fn_psi, in_axes=(0, 0, 0, None))
#   vectorized_update_chain_phi = jax.vmap(update_chain_fn_phi, in_axes=(0, 0, 0, None))

#   # Equilibrate psi and phi chains
#   new_batch_psi_configs, new_batch_psis, num_accepts_psi = vectorized_update_chain_psi(rngs_psi, psi_configs_burn, psis_burn, w_psi) 
#   new_batch_phi_configs, new_batch_phis, num_accepts_phi = vectorized_update_chain_phi(rngs_phi, phi_configs_burn, phis_burn, w_phi) 

#   # Compute psi ratios
#   psi_ratio_fn = functools.partial(psi_ratio, w2=w_phi, phi=phi)
#   psi_ratio_fn_vec = jax.vmap(psi_ratio_fn, in_axes=(0, 0))
#   psi_ratio_array = psi_ratio_fn_vec(new_batch_psi_configs, new_batch_psis)

#   # Compute psi sq ratios
#   # psi_ratio_mod_sq_fn_2 = functools.partial(psi_ratio_mod_sq, w2=w_psi, phi=psi)
#   # psi_ratio_mod_sq_fn_vec_2 = jax.vmap(psi_ratio_mod_sq_fn, in_axes=(0, 0))
#   # psi_ratio_mod_sq_array = psi_ratio_mod_sq_fn_vec(new_batch_phi_configs, new_batch_phis)
#   psi_ratio_fn_2 = functools.partial(psi_ratio, w2=w_psi, phi=psi)
#   psi_ratio_fn_vec_2 = jax.vmap(psi_ratio_fn_2, in_axes=(0, 0))
#   psi_ratio_array_2 = psi_ratio_fn_vec_2(new_batch_psi_configs, new_batch_psis)  
#   # Use a more symmetric formula
#   # psi_ratio_mod_sq_array = psi_ratio_fn_vec_2(new_batch_phi_configs, new_batch_phis)

#   # Compute mean from sampled configs
#   psi_ratio_mean = jnp.mean(psi_ratio_array, axis=0)
#   psi_ratio_var = jnp.std(psi_ratio_array, axis=0)
#   # psi_ratio_mod_sq_mean = jnp.mean(psi_ratio_mod_sq_array, axis=0)
#   psi_ratio_mean_2 = jnp.mean(psi_ratio_array_2, axis=0)
#   psi_ratio_var_2 = jnp.std(psi_ratio_array_2, axis=0)

#   # return psi_ratio_mean * jnp.sqrt(psi_ratio_mod_sq_mean)
#   std_all = psi_ratio_var * psi_ratio_var_2 + psi_ratio_var * psi_ratio_mean_2**2 + psi_ratio_var_2 * psi_ratio_mean**2 
#   return jnp.sqrt(psi_ratio_mean * jnp.conjugate(psi_ratio_mean_2)), std_all / np.sqrt(num_batch_psi)
