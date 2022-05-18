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
import xarray as xr
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
DataKey = collections.namedtuple('DataKey', ['sector', 'h_field', 'iteration'])

##### to-do change this!
def generate_field_noise_batch(key, 
                               params, 
                               noise_amp, 
                               batch, shape, 
                               return_noise=False):
  # rbm_exp_params = tc_utils.convert_rbm_expanded(params, (shape[0]//2, shape[1]))
  rngs = utils.split_key(key, (batch, 2))
  rbm_noise_params_batch = [tc_utils.generate_uniform_noise_param(rngs[i], params,  
                                                                  noise_amp, return_noise) for i in range(batch)]
  if return_noise:
    return list(map(list, zip(*rbm_noise_params_batch)))                                                                  
  return rbm_noise_params_batch

def generate_samples_T(key, h_field, spin_shape, 
                       len_chain_E, burn_E_factor, num_samples_E, 
                       psi_apply, 
                       len_chain, burn_factor, num_samples, 
                       T, noise_amp, 
                       p_flip,
                       params, 
                       angle, 
                       return_results=False, 
                       init_noise=0.):
  num_spins = spin_shape[0] * spin_shape[1]
  # Parameter for estimating energy
  len_chain_burn_E = burn_E_factor * num_spins
  # Parameters for mcmc_param
  len_chain_burn= burn_factor
  ham = tc_utils.set_up_ham_field_rotated(spin_shape, h_field, angle)
  estimate_ET_fn = functools.partial(estimates_mcmc.estimate_operator, operator=ham, psi_apply=psi_apply, 
                                    len_chain=len_chain_E, len_chain_first_burn=len_chain_burn_E, spin_shape=spin_shape, 
                                    num_samples=num_samples_E)
  key_sample, key_init, key_flip = jax.random.split(key, 3)
  rbm_noise_params_batch = generate_field_noise_batch(key_init, params,
                                                      init_noise, num_samples, spin_shape)
  # propose_move_param_fn = functools.partial(tc_utils.generate_FV_noise_param, amp_noise=noise_amp)
  propose_move_param_fn = functools.partial(tc_utils.propose_param_fn, p_mpar=p_flip, amp_noise=noise_amp)
  accept_rate, new_samples_dict, new_energies = mcmc_param.energy_sampling_mcmc(key_sample, rbm_noise_params_batch, 
                                                                  len_chain_burn, len_chain, estimate_ET_fn, 
                                                                  propose_move_param_fn, T, 
                                                                   return_results=return_results)

  new_samples = jax.tree_map(lambda x: einops.rearrange(x, ' a b c d e -> (a b) c d e'), new_samples_dict )
  new_energies = einops.rearrange(new_energies, ' a b -> (a b)') / num_spins
  return accept_rate, new_samples, new_energies    

def main(argv):
  config = FLAGS.config
  job_id = config.job_id
  file_id = config.file_id
  h_field, T = config.h_and_t[file_id]
  sector_list = config.sector_list
  rng_seq = hk.PRNGSequence(42 + file_id)
  data_dir = config.data_dir
  output_dir = config.output_dir
  iteration = config.iter
  spin_shape = config.spin_shape
  num_spins = spin_shape[0] * spin_shape[1]
  angle = config.angle
  noise_amp = config.noise_amp
  p_mparticle = config.p_mparticle
  # Parameters for mcmc energy
  len_chain_E = config.len_chain_E
  burn_E_factor = config.burn_E_factor
  num_samples_E = config.num_samples_E
  # Parameters for mcmc_param
  len_chain = config.len_chain
  burn_factor= None
  num_samples = config.num_samples
  filename_ens = config.filenames_ens[file_id]
  filename_ens_property = config.filenames_ens_property[file_id]

  num_sec = len(sector_list)
  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_noise))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # jit function
  generate_samples_T_fn = functools.partial(
      generate_samples_T, spin_shape=spin_shape, psi_apply=psi_apply,
      len_chain_E=len_chain_E, burn_E_factor=burn_E_factor, num_samples_E=num_samples_E, 
      len_chain=len_chain, burn_factor=burn_factor, num_samples=num_samples, 
      noise_amp=noise_amp, p_flip=p_mparticle)
  generate_samples_T_jit = jax.jit(generate_samples_T_fn, static_argnums=(8, ))

  new_samples_list = []
  new_energies_list = []
  accept_rate_list = []
  
  params_dict = pickle.load(open(data_dir + "params_dict.p", "rb"))
  params_list = [tc_utils.convert_rbm_expanded(params_dict[(sec, h_field, iteration)], (spin_shape[0]//2, spin_shape[1])) for sec in range(1, 5)]
  for params in tqdm(params_list, desc="param sector"):
    accept_rate, new_samples, new_energies = generate_samples_T_jit(key=next(rng_seq), h_field=h_field, T=T, params=params, angle=angle)      
    new_samples_list.append(new_samples)
    new_energies_list.append(new_energies)
    accept_rate_list.append(accept_rate)
  samples_all_secs = utils.concat_along_axis(new_samples_list, axis=0)
  energies_all_secs = np.concatenate(new_energies_list)[np.newaxis, np.newaxis, np.newaxis, ...] # numpy should work here because only generate_smaples is jitted
  accepts_all_secs = np.concatenate(accept_rate_list)[np.newaxis, np.newaxis, np.newaxis, ...]    
  data_vars = {}
  data_vars['energy'] = (["h", "T", "iter", "samples"], energies_all_secs)
  data_vars['accept'] = (["h", "T", "iter", "chains"], accepts_all_secs)
  my_dataset = xr.Dataset(data_vars=data_vars, coords=dict(h=[h_field], T=[T], iter=[iteration], 
    samples=np.arange(len_chain * num_samples * num_sec), chains=np.arange(num_samples * num_sec)))
  # save output
  pickle.dump(samples_all_secs, open(output_dir + filename_ens, 'wb'))
  my_dataset.to_netcdf(path=output_dir + filename_ens_property)
  with open(output_dir+f'config_{job_id}.json', 'w') as f:
    json.dump(config.to_json(), f)  

if __name__ == '__main__':
  app.run(main)
