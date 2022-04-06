import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp
import functools
import pickle
import importlib
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import seaborn
import xarray as xr
import scipy
from scipy.interpolate import griddata
from tqdm import tqdm
import itertools
import pandas as pd
import einops
import os.path
import math

#@title main fixed angle
def main_no_carry_angle_flexible(h_field_array, epsilon, spin_shape, num_chains, num_steps, first_burn_len, 
                        len_chain, learning_rate, spin_flip_p, angle, regulation, 
                        main_key, sector=None, Jf=1., model_name=None, epsilon_carry=0.3):
  # Full optimization for each field value
  def _update_fields(carry, inputs):
    # Initialize params and configs, psis from previous field
    carried_params, _, _ = carry
    h_field, key = inputs
    equi_key, init_key, carried_noise_key = jax.random.split(key, 3)
    carried_params = tc_utils.generate_uniform_noise_param(carried_noise_key, carried_params, epsilon_carry)
    init_configs = sample_utils.init_samples(init_key, num_spins, num_chains) 
    init_psis = psi_apply_vectorized(carried_params, init_configs)
    #Optimizer initialization
    opt_init, opt_update = optax.adam(learning_rate)
    opt_state = opt_init(carried_params)
    # Define hamiltonian 
    myham = tc_utils.set_up_ham_field_rotated(spin_shape, h_field, angle, Jf=Jf)
    # First burn update chain function
    update_chain_fn_first_burn = functools.partial(mcmc.update_chain,        #vmap update_chain for first burn
                                        len_chain=first_burn_len, psi=psi_apply, 
                                        propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample, 
                                        )
    update_chain_first_burn_vec = jax.vmap(update_chain_fn_first_burn, in_axes=(0, 0, 0, None))
    # Equilibrated configs and psis
    new_key, sub_key = jax.random.split(equi_key)
    rngs = utils.split_key(new_key, np.array([num_chains, 2]))
    # # Check without chaining the configurations for increasing field
    # fixed_configs = init_configs
    # fixed_config_psis = psi_apply_vectorized(carried_params, fixed_configs)
    # equilibrated_configs, equilibrated_psis, __ = update_chain_first_burn_vec(rngs, fixed_configs, fixed_config_psis, carried_params)
    # print(equilibrated_configs.shape)
    equilibrated_configs, equilibrated_psis, __ = update_chain_first_burn_vec(rngs, init_configs, init_psis, carried_params)

    # Define partial MCMC_optimization fn
    MCMC_optimization_fn = functools.partial(optimizations.MCMC_optimization, psi=psi_apply, init_params=carried_params, opt_update=opt_update, init_opt_state=opt_state, num_steps=num_steps, 
                                             len_chain=len_chain, propose_move_fn=propose_move_fn, make_move_fn=sample_utils.vertex_bond_sample, ham=myham,
                                                                                                              learning_rate=learning_rate, regulation=regulation)
    #compile with jit
    MCMC_optimization_jit = jax.jit(MCMC_optimization_fn)

    ((new_batch_configs, new_batch_psis, new_model_params, new_opt_state), 
      (num_accepts, energy_expectation, grad_psi_expectation, grad_energy_expectation)) = MCMC_optimization_jit(sub_key, equilibrated_configs, equilibrated_psis )
    return (new_model_params, new_batch_configs, new_batch_psis), energy_expectation[:, 0], num_accepts, grad_energy_expectation

  noise_key, init_config_key, field_key = jax.random.split(main_key, 3)
  num_spins = spin_shape[0] * spin_shape[1]

  if model_name == 'rbm_noise':
    model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_noise))
  elif model_name == 'rbm': 
    model = hk.without_apply_rng(hk.transform(wavefunctions.fwd))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)
  psi_apply_vectorized = jax.vmap(psi_apply, in_axes=(None, 0))     #vmap psi  

  vertex_bonds = tc_utils.get_vertex_bonds(spin_shape)
  propose_move_fn = functools.partial(mcmc.propose_move_fn, p=spin_flip_p, vertex_bonds=vertex_bonds)
  h_field_length = h_field_array.shape[0]
  # Create key for all field values
  rngs = jax.random.split(field_key, h_field_length)      #Split the keys based on batch size and steps

  # List for saving parameters and energies
  new_params_list = []
  energy_density_list = []
  energy_steps_list = []
  psis_list = []
  num_accepts_list = []
  grad_list = []
  
  init_configs = sample_utils.init_samples(init_config_key, num_spins, num_chains)     #Create intial chains
  if sector == None:
    params = model.init(noise_key, spin_shape=spin_shape, x=init_configs[0,...])
  else:
    rbm_params = tc_utils.get_rbm_params(sector)      # Get initial parameter for sector
    if model_name == 'rbm_noise':
      rbm_params = tc_utils.convert_rbm_expanded(rbm_params, (spin_shape[0]//2, spin_shape[1]))
    params = tc_utils.generate_uniform_noise_param(noise_key, rbm_params, epsilon)
    # params = tc_utils.set_partial_params_const(params, ['wV', 'bV'], 0., model_name=model_name)
  print(f"Initial parameters are")
  fig, axs = plt.subplots(1, 2, figsize=(8 * 5, 4 ))
  if model_name == 'rbm_noise':
    plot_utils.plot_weights_noise(axs, params, h_field_array[0], 'rbm_noise')
    plt.show()
  elif model_name == 'rbm':
    plot_utils.plot_weights(axs, params, h_field_array[0], 'rbm', )
    plt.show()
  new_params = params
  new_configs = init_configs
  new_psis = psi_apply_vectorized(new_params, new_configs)
  
  for i, h_field in tqdm(enumerate(h_field_array)):
    # print(f"{i}/{h_field_length}-th iteration at h={h_field} ")
    (updated_params, updated_configs, updated_psis), energy_steps, num_accepts, grad_energy_expectation = _update_fields((new_params, new_configs, new_psis), 
                                                                                                          (h_field, rngs[i, :]))
    new_params = updated_params
    new_configs = updated_configs
    new_psis = updated_psis
    new_params_list.append(new_params)
    energy_density_list.append(energy_steps[-1] / num_spins)
    energy_steps_list.append(energy_steps / num_spins)
    psis_list.append(updated_psis)
    num_accepts_list.append(num_accepts)
    grad_list.append(grad_energy_expectation)

    print(f"Current energy at h={h_field} is {energy_steps[-1] / num_spins}")
  return new_params_list, energy_density_list, updated_psis, energy_steps_list, psis_list,num_accepts_list, grad_list, params