#import opt_utils
import re
import datetime

import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp
import functools
import pickle
import importlib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import scipy
from scipy.interpolate import griddata
from tqdm import tqdm
import itertools
import einops
import math
from absl import app
import json
import sys
import os

# setting path
sys.path.append('..')
import wavefunctions
import operators
import bonds
import utils
import tc_utils
import train_utils
import sample_utils
import mcmc
import optimizations
import overlaps
import diffusion_map
import estimates_mcmc
import mcmc_param
import exact_comp

from absl import flags
import collections
import ml_collections
from ml_collections.config_flags import config_flags

config_flags.DEFINE_config_file('config')
FLAGS = flags.FLAGS

#@title main fixed angle
def _optimize_over_fields(h_field_array, epsilon, spin_shape, num_chains, num_steps, first_burn_len,
                        len_chain, learning_rate, spin_flip_p, angle,
                        main_key, sector=None, Jf=1., model_name=None, epsilon_carry=0.):
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
                                                                                                              learning_rate=learning_rate)
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
  elif model_name == 'rbm_cnn':
    model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_cnn))
  elif model_name == 'rbm_cnn_2':
    model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_cnn_2))
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
  exact_energy_list = []

  init_configs = sample_utils.init_samples(init_config_key, num_spins, num_chains)     #Create intial chains
  if sector == None:
    params = model.init(noise_key, spin_shape=spin_shape, x=init_configs[0,...])
  else:
    if model_name == 'rbm':
      my_params = tc_utils.get_rbm_params(sector)      # Get initial parameter for sector
    elif model_name == 'rbm_noise':
      rbm_params = tc_utils.get_rbm_params(sector)      # Get initial parameter for sector
      my_params = tc_utils.convert_rbm_expanded(rbm_params, (spin_shape[0]//2, spin_shape[1]))
    elif model_name == 'rbm_cnn':
      my_params = tc_utils.get_cnn_params(sector)
    elif model_name == 'rbm_cnn_2':
      my_params = tc_utils.get_cnn_channel_params(sector, channel=2)
    else:
      # todo: this line needs to be modified
      _, noise_key_2 = jax.random.split(noise_key, 2)
      my_params = model.init(noise_key_2, spin_shape=spin_shape, x=init_configs[0,...])
    params = tc_utils.generate_uniform_noise_param(noise_key, my_params, epsilon)

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
    myham = tc_utils.set_up_ham_field_rotated(spin_shape, h_field, angle, Jf=Jf)
    exact_energy = exact_comp.compute_op_fn(new_params, psi_apply, myham, num_spins, batch_size=64)
    new_params_list.append(new_params)
    energy_density_list.append(energy_steps[-1] / num_spins)
    energy_steps_list.append(energy_steps / num_spins)
    psis_list.append(updated_psis)
    num_accepts_list.append(num_accepts)
    grad_list.append(grad_energy_expectation)
    exact_energy_list.append(exact_energy / num_spins)

    print(f"Current energy at h={h_field} is {energy_steps[-1] / num_spins}")
  return new_params_list, energy_density_list, exact_energy_list, energy_steps_list, num_accepts_list, psis_list, grad_list

def main(argv):
  print(f'Program has started with args: {argv}')

  config = FLAGS.config
  file_id = config.file_id
  output_dir = config.data_dir
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  job_id = config.job_id
  filename_save = config.filenames_save[file_id]
  model_name = config.model_name
  # Parameters for mcmc energy
  num_steps = config.num_steps
  num_chains = config.num_chains
  learning_rate = config.learning_rate
  spin_flip_p = config.spin_flip_p
  burn_in_factor = config.burn_in_factor
  sector, iteration = config.sec_iter[file_id]
  spin_shape = config.spin_shape
  h_params = config.h_params

  h_field_array = np.round(np.array(h_params), 3)
  angle = config.angle
  epsilon = config.epsilon
  num_spins = spin_shape[0] * spin_shape[1]
  rng_seq = hk.PRNGSequence(42 + sector * iteration)
  params_list_list = []
  energies_list = []
  energy_steps_list = []
  init_param_list = []
  all_results_list = []
  main_key = next(rng_seq)
  results = _optimize_over_fields(h_field_array=h_field_array, epsilon=epsilon, spin_shape=spin_shape, num_chains=num_chains, num_steps=num_steps,
                                                            first_burn_len=num_spins*burn_in_factor, len_chain=30, learning_rate=learning_rate, spin_flip_p=spin_flip_p, main_key=main_key,
                                                            angle=angle, model_name=model_name, sector=sector)


  # h_field_list = list(h_field_array)
  field_results_dict = [dict(zip(h_field_array, data)) for data in results]
  pickle.dump(field_results_dict, open(output_dir + filename_save, 'wb'))
  with open(output_dir+f'config_{job_id}.json', 'w') as f:
    json.dump(config.to_json(), f)  

if __name__ == '__main__':
  app.run(main)
