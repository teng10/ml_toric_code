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
# flags.DEFINE_integer('job_id', 0, 'slurm job id.')
FLAGS = flags.FLAGS

def get_vector(num_sites, batch_size, psi, psi_params):
  """Generates a full wavefunction by evaluating `psi` on basis elements."""
  # print(basis_iterator)
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
       
  psi_fn = jax.jit(jax.vmap(functools.partial(psi, psi_params)))
  psi_values = []
  basis_iterator = _get_full_basis_iterator(num_sites, batch_size)
  for cs in basis_iterator:
    psi_values.append(jax.device_get(psi_fn(cs)))
  return np.concatenate(psi_values)  

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

def main(argv): 
  #Load parameters from config file
  config = FLAGS.config
  spin_shape=(6, 3)
  num_spins = spin_shape[0] * spin_shape[1]
  batch_size = 64
  # iteration = config.iter
  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_noise))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)  
  rng_seq = hk.PRNGSequence(43)
  data_dir = config.data_dir
  output_dir = os.path.join(data_dir, "DMdata/")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  job_id = config.job_id
  (h, T, iteration) = config.h_t_iter[config.file_id]
  filename_load = config.filenames_load[config.file_id]
  filename_save = config.filenames_save[config.file_id]
  
  # Load data 
  params_loaded = pickle.load(open(data_dir + filename_load, "rb"))
  data_size = jax.tree_leaves(utils.shape_structure(params_loaded))[0]    #Size of the data
  ens_idx_array = np.arange(data_size)
  #Compute vectors exactly
  vectors = []
  for i in range(data_size):
    param = utils.slice_along_axis(params_loaded, 0, i)
    vec = get_vector(num_spins, batch_size, psi_apply, param)
    vectors.append(vec)
  #Compute overlap matrix given `vectors`
  overlap_mat = _get_overlap_matrix(vectors)
  overlap_mat = overlap_mat[np.newaxis, np.newaxis, np.newaxis, ...]
  data_vars = {}
  data_vars['overlap'] = (['h', 'T', 'iter', 'ens_idx_1', 'ens_idx_2'], overlap_mat)
  ds = xr.Dataset(data_vars=data_vars, coords=dict(h=[h], T=[T], iter=[iteration], 
    ens_idx_1=ens_idx_array, ens_idx_2=ens_idx_array))

  ds.to_netcdf(path=output_dir+filename_save)
  # Save config file as json
  with open(f'config_{job_id}.json', 'w') as f:
    json.dump(output_dir+config.to_json(), f)

if __name__ == '__main__':
  app.run(main)