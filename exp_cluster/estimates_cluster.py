#@title Estimating operators in a dictionary

import re
import datetime

import numpy as np
import haiku as hk
#import optax
import jax
import jax.numpy as jnp
import functools
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
#import seaborn
import xarray as xr
import scipy
from scipy.interpolate import griddata
from tqdm import tqdm
import itertools
import einops
import math
from absl import app
import collections

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
import overlaps
import diffusion_map
import estimates_mcmc
import mcmc_param
import exact_comp

def _get_op_dict(directions, loop_indices, spin_shape, h, angle):
  # Build operator dictionary
  num_spins = spin_shape[0] * spin_shape[1]
  op_dict = {}
  ham = tc_utils.set_up_ham_field_rotated(spin_shape, h, angle) # Build hamiltonian operator
  op_dict['ham'] = ham
  pauliZ = operators.PauliBondZ(hz=1., bond=np.arange(num_spins))
  op_dict['pauliZ'] = pauliZ
  for dir in directions:  # Build sets of wilson loop operators
    for idx in loop_indices:
      label = [f'WLX{idx}', f'WLY{idx}']
      loop_x = operators.WilsonLXBond(bond=bonds.wilson_loops(spin_shape, dir, idx))
      op_dict[label[dir]] = loop_x
  return op_dict

DataKey = collections.namedtuple('DataKey', ['sector', 'h_field', 'iteration'])

def main(argv): 
  spin_shape=(6, 3)
  num_spins = spin_shape[0] * spin_shape[1]
  angle = 0.
  rng_seq = hk.PRNGSequence(43)
  noise_amp = 0.2
  p_mparticle = 0.3
  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)
  # Parameters for mcmc energy
  len_chain_E=30
  burn_E_factor = 500
  num_samples_E = 300
  sec_list = [1, 2, 3, 4]
  iteration_list = [int(argv[1])]

  file_path = argv[2]
  params_loaded = pickle.load(open(file_path + "params_dict.p", "rb"))
  # complete this
  # h_list = np.array() 
  h_field_array = np.array(list(sorted(set([list(params_loaded.keys())[i].h_field for i in range(len(params_loaded.keys()))]))))
      
  # Parameters for mcmc energy
  num_spins = spin_shape[0] * spin_shape[1]
  len_chain_burn_E = burn_E_factor * num_spins
  directions = [0, 1]
  loop_indices = range(spin_shape[1])

  file_path_mcmc = file_path + "estimates_mcmc/"
  results_desc = ["MCMC_ev", "MCMC_std", "MCMC_local_vals", "MCMC_accept_mean"]
  op_dict_desc = "WL_H_S"
  file_name = f"_{op_dict_desc}_{spin_shape}_{iteration_list[0]}.nc"
  full_data = []
  for sec in sec_list:
    sec_data = []
    for h in tqdm(h_field_array, desc="h"):
      h_data = []
      for i in iteration_list:
        param = params_loaded[(sec, h, i)]
        op_dict = _get_op_dict(directions, loop_indices, spin_shape, h, angle)
        # Define the jitted estimation function for a set of operators
        estimate_op_dict_fn = functools.partial(estimates_mcmc.estimate_operator_dict, operator_dict=op_dict, psi_apply=psi_apply, 
                                        len_chain=len_chain_E, len_chain_first_burn=len_chain_burn_E, spin_shape=spin_shape, 
                                        num_samples=num_samples_E)
        # estimate_op_dict_vec = jax.vmap(estimate_op_dict_fn, in_axes=(0, 0))
        estimate_op_dict_jit = jax.jit(estimate_op_dict_fn)
        # MCMC estimates
        rng_key = next(rng_seq)
        results = estimate_op_dict_jit(rng_key, param)
        h_data.append(results)
      sec_data.append(jax.tree_map(lambda *args: np.stack(args), *h_data))
    full_data.append(jax.tree_map(lambda *args: np.stack(args), *sec_data))
  results_tuple = jax.tree_map(lambda *args: np.stack(args), *full_data)
  data_vars = {}
  data_vars['acceptance'] = (['sec', 'h', 'iter'], results_tuple[-1])
  data_attrs = ['ev_', 'std_']
  for data, desc in zip(results_tuple[:-2], data_attrs):
    for k, val in data.items():
      data_vars[desc+k] = (['sec', 'h', 'iter'], val)
  for k, val in results_tuple[2].items():
    data_vars['local_'+k] = (['sec', 'h', 'iter', 'batch'], val)   
  my_dataset = xr.Dataset(data_vars=data_vars, coords=dict(sec=sec_list, h=h_field_array, iter=iteration_list))
  my_dataset.to_netcdf(path=file_path_mcmc+file_name)

if __name__ == '__main__':
  app.run(main)