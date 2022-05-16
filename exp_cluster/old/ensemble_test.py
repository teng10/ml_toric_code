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
#from sklearn.cluster import KMeans
#from sklearn.decomposition import KernelPCA
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
#import seaborn
#import xarray as xr
import scipy
from scipy.interpolate import griddata
from tqdm import tqdm
import itertools
#import pandas as pd
import einops
#import os.path
import math
from absl import app

import sys

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
#import plot_utils
import overlaps
import diffusion_map
import estimates_mcmc
import mcmc_param
import exact_comp

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
  #
  config = FLAGS.config
  brun_energy_factor = config.burn_energy_factor
  
  #
  print(f'Program has started with args: {argv}')
  h_field = float(argv[1])
  T = float(argv[2])
  spin_shape=(6, 3)
  num_spins = spin_shape[0] * spin_shape[1]
  angle = 0.
  rng_seq = hk.PRNGSequence(42 + int(float(argv[3])))
  noise_amp = 0.2
  p_mparticle = 0.3
  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_noise))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)
  # Parameters for mcmc energy
  len_chain_E=30
  burn_E_factor = 500
  num_samples_E = 200
  # Parameters for mcmc_param
  len_chain= 300
  burn_factor= None
  num_samples=5
  # sector_labels = np.concatenate(([np.full((num_samples * len_chain,), i) for i in sector_list]))     # Create labels for each batch of samples 
  file_path = argv[4]
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
  screened_params_dict = pickle.load(open(file_path+"../../"+"params_screened_temp.p", "rb"))
  params_list = utils.split_axis(screened_params_dict[h_field], 0)
  for params in tqdm(params_list, desc="param sector"):
    accept_rate, new_samples, new_energies = generate_samples_T_jit(key=next(rng_seq), h_field=h_field, T=T, params=params, angle=angle)      
    new_samples_list.append(new_samples)
    new_energies_list.append(new_energies)
    accept_rate_list.append(accept_rate)
  file_name_samples = f"samples_{spin_shape}_hz{h_field}_T{T}.p"
  file_name_energies = f"energies_{spin_shape}_hz{h_field}_T{T}.p"
  file_name_accept_rate = f"accepts_{spin_shape}_hz{h_field}_T{T}.p"
  samples_all_secs = utils.concat_along_axis(new_samples_list, axis=0)
  energies_all_secs = jnp.concatenate(new_energies_list)
  accepts_all_secs = jnp.concatenate(accept_rate_list)    

  now = datetime.datetime.now()
  pattern = re.compile(r"-\d\d-\d\d")
  mo = pattern.search(str(now))
  date = mo.group()[1:]

  pickle.dump(samples_all_secs, open(file_path+file_name_samples, 'wb'))
  pickle.dump(energies_all_secs, open(file_path+file_name_energies, 'wb'))
  pickle.dump(accepts_all_secs, open(file_path+file_name_accept_rate, 'wb'))

if __name__ == '__main__':
  app.run(main)
