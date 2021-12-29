import jax
import jax.numpy as jnp
import numpy as np
import functools
import utils
import sample_utils
import tc_utils
import train_utils
import eval_utils
import mcmc
import bonds
import operators

def estimate_energy(key, 
                    param,   
                    h_field, 
                    psi_apply, 
                    len_chain, 
                    len_chain_first_burn,
                    spin_shape, 
                    num_samples,
                    propose_move_fn=mcmc.propose_move_fn, 
                    make_move_fn=sample_utils.vertex_bond_sample, 
                    p=0.4):
  """
  Return <E> via mcmc sampling. Inlcude first burn for equalibriating samples. 
  """
  num_spins = spin_shape[0] * spin_shape[1]
  new_key, sub_key = jax.random.split(key, 2)
  # Generate initial configs for mcmc
  init_configs = sample_utils.init_samples(new_key, num_spins, num_samples)
  psi_apply_vec = jax.vmap(psi_apply, in_axes=(None, 0))
  init_psis = psi_apply_vec(param, init_configs)
  # Set up hamiltonian
  ham = tc_utils.set_up_ham_field(spin_shape, h_field)
  # Define propose move fn
  vertex_bonds = tc_utils.get_vertex_bonds(spin_shape)
  propose_move_fn = functools.partial(mcmc.propose_move_fn, vertex_bonds=vertex_bonds)    

  # First burn update chain function
  update_chain_fn = functools.partial(mcmc.update_chain,        #vmap update_chain for first burn
                                      psi=psi_apply, 
                                      propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample, 
                                      p=p)
  update_chain_vectorized = jax.vmap(update_chain_fn, in_axes=(0, 0, 0, None, None))
  # First burn for configs and psis
  key1, key2 = jax.random.split(sub_key, 2)
  rngs_first_burn = utils.split_key(key1, (num_samples, 2))
  rngs = utils.split_key(key2, (num_samples, 2))
  first_burn_configs, first_burn_psis, num_accepts = update_chain_vectorized(rngs_first_burn, 
                                                    init_configs, init_psis, param, len_chain_first_burn)  

  #Equilibrate chains   
  new_batch_configs, new_batch_psis, num_accepts = update_chain_vectorized(rngs, 
                                            first_burn_configs, first_burn_psis, param, len_chain)                                              
  # Compute expectation values using equilibriated cs and psis                                                    
  _, energy_ev, _, _ = train_utils.grad_energy_expectation_gradbatch_fn(new_batch_configs, new_batch_psis, 
                                                                                               psi_apply, param, ham)
  return energy_ev[0]

def estimate_operator(key, 
                    param,   
                    operator, 
                    psi_apply, 
                    len_chain, 
                    len_chain_first_burn,
                    spin_shape, 
                    num_samples,
                    propose_move_fn=mcmc.propose_move_fn, 
                    make_move_fn=sample_utils.vertex_bond_sample, 
                    p_spinflips=0.4, 
                    return_psi_cs=False):
  """
  Return <E> via mcmc sampling. Inlcude first burn for equalibriating samples. 
  """
  num_spins = spin_shape[0] * spin_shape[1]
  new_key, sub_key = jax.random.split(key, 2)
  # Generate initial configs for mcmc
  init_configs = sample_utils.init_samples(new_key, num_spins, num_samples)
  psi_apply_vec = jax.vmap(psi_apply, in_axes=(None, 0))
  init_psis = psi_apply_vec(param, init_configs)
  # # Set up hamiltonian
  # ham = tc_utils.set_up_ham_field(spin_shape, h_field)
  # wilson_loop = operators.WilsonLXBond(bond=bonds.get_wilson_loop(spin_shape, direction))
  # Define propose move fn
  vertex_bonds = tc_utils.get_vertex_bonds(spin_shape)
  propose_move_fn = functools.partial(mcmc.propose_move_fn, vertex_bonds=vertex_bonds, p=p_spinflips)    

  # First burn update chain function
  update_chain_fn = functools.partial(mcmc.update_chain,        #vmap update_chain for first burn
                                      psi=psi_apply, 
                                      propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample)
  update_chain_vectorized = jax.vmap(update_chain_fn, in_axes=(0, 0, 0, None, None))
  # First burn for configs and psis
  key1, key2 = jax.random.split(sub_key, 2)
  rngs_first_burn = utils.split_key(key1, (num_samples, 2))
  rngs = utils.split_key(key2, (num_samples, 2))
  first_burn_configs, first_burn_psis, num_accepts = update_chain_vectorized(rngs_first_burn, 
                                                    init_configs, init_psis, param, len_chain_first_burn)  

  #Equilibrate chains   
  new_batch_configs, new_batch_psis, num_accepts = update_chain_vectorized(rngs, 
                                            first_burn_configs, first_burn_psis, param, len_chain)                                                                                      
  # Compute expectation values using equilibriated cs and psis                                                    
  op_ev, op_std, op_local_batch = eval_utils.op_expectation_fn(new_batch_configs, new_batch_psis, psi_apply, param, operator)
  if return_psi_cs:
    return op_ev, op_std, jnp.mean(num_accepts/len_chain), op_local_batch, new_batch_configs, new_batch_psis,
  return op_ev, op_std, jnp.mean(num_accepts/len_chain), op_local_batch