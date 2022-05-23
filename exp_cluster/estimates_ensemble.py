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
import os
import json

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
flags.DEFINE_string('output_path', 'tmp.txt', 'Path to which save the data.')
flags.DEFINE_integer('file_id', 0, 'data file id.')
flags.DEFINE_integer('job_id', 0, 'slurm job id.')
FLAGS = flags.FLAGS

def _get_op_dict(directions, loop_indices, spin_shape, h, angle):
  # Build operator dictionary
  num_spins = spin_shape[0] * spin_shape[1]
  op_dict = {}
  ham = tc_utils.set_up_ham_field_rotated(spin_shape, h, angle) # Build hamiltonian operator
  op_dict['ham'] = ham
  # pauliZ = operators.PauliBondZ(hz=1., bond=np.arange(num_spins))
  pauliZ =  operators.ToricCodeHamiltonianRotated(Jv=0., Jf=0., h=1., hx=0., face_bonds=[], vertex_bonds=[], pauli_bonds=np.arange(0, num_spins, 1))  
  op_dict['pauliZ'] = pauliZ
  for dir in directions:  # Build sets of wilson loop operators
    for idx in loop_indices:
      label = [f'WLX{idx}', f'WLY{idx}']
      loop_x = operators.WilsonLXBond(bond=bonds.wilson_loops(spin_shape, dir, idx))
      op_dict[label[dir]] = loop_x
  return op_dict

DataKey = collections.namedtuple('DataKey', ['sector', 'h_field', 'iteration'])

def main(argv): 
  config = FLAGS.config
  spin_shape=(6, 3)
  num_spins = spin_shape[0] * spin_shape[1]
  angle = 0.
  rng_seq = hk.PRNGSequence(43)
  noise_amp = 0.2
  p_mparticle = 0.3
  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_noise))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)
  # Parameters for mcmc energy
  len_chain_E = config.len_chain_E
  burn_E_factor = config.burn_E_factor
  num_samples_E = config.num_samples_E
  data_dir = config.data_dir
  output_dir = os.path.join(data_dir, "estimates/")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  job_id = config.job_id
  (h, T, iteration) = config.h_t_iter[config.file_id]
  filename_load = config.filenames_load[config.file_id]
  filename_save = config.filenames_save[config.file_id] + f"_id_{config.job_id}.nc"

  
  params_loaded = pickle.load(open(data_dir + filename_load, "rb"))
  data_size = jax.tree_leaves(utils.shape_structure(params_loaded))[0]
  all_indices = np.arange(data_size)
  data_indices = all_indices[slice(job_id, data_size, config.num_workers)]
      
  # Parameters for mcmc energy
  num_spins = spin_shape[0] * spin_shape[1]
  len_chain_burn_E = burn_E_factor * num_spins
  directions = [0, 1]
  loop_indices = range(spin_shape[1])
  
  results_desc = ["ev_", "std_", "local_", "accept"]
  
  idx_data = []
  for idx in data_indices:
    param = utils.slice_along_axis(params_loaded, 0, idx)
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
    idx_data.append(results)

  results_tuple = jax.tree_map(lambda *args: np.stack(args), *idx_data)
  # we expand first 3 dims to represent fixed dimension `h`, `T` and `iter`.
  results_tuple = jax.tree_map(lambda x: x[np.newaxis, np.newaxis, np.newaxis, ...], results_tuple)
  data_vars = {}
  data_vars[results_desc[-1]] = (['h', 'T', 'iter', 'ensemble_id'], results_tuple[-1])
  data_attrs = results_desc[:2]
  for data, desc in zip(results_tuple[:-2], data_attrs):
    for k, val in data.items():
      data_vars[desc+k] = (['h', 'T', 'iter', 'ensemble_id'], val)
  # for k, val in results_tuple[2].items():
  #   data_vars['local_'+k] = (['h', 'T', 'iter', 'ensemble_id', 'batch'], val)   
  my_dataset = xr.Dataset(data_vars=data_vars, coords=dict(h=[h], T=[T], iter=[iteration], ensemble_id=data_indices))
  my_dataset.to_netcdf(path=output_dir+filename_save)
  with open(output_dir+f'config_{job_id}.json', 'w') as f:
    json.dump(config.to_json(), f)

if __name__ == '__main__':
  app.run(main)