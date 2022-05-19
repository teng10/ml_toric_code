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

# def _get_similarity_matrix(similarity_fn, params_stacked):
#   # params_list = utils.split_axis(params_stacked, axis=0)
#   num_params = jax.tree_leaves(utils.shape_structure(params_stacked))[0]
#   s_mat = np.zeros((num_params, num_params))
#   for i in range(num_params):
#     for j in range(i, num_params):
#       param1 = utils.slice_along_axis(params_stacked, 0, i)
#       param2 = utils.slice_along_axis(params_stacked, 0, j)
#       sim = similarity_fn(param1, param2)
#       s_mat[i, j] = sim
#       s_mat[j, i] = sim
#   return s_mat
def _get_similarity_matrix(similarity_fn, params_stacked):
  S_vec = jax.vmap(similarity_fn, in_axes=(0, None))
  S_vec_vec = jax.vmap(S_vec, in_axes=(None, 0))
  S_mat = S_vec_vec(params_stacked, params_stacked)
  return  jax.device_get(S_mat)
  
def main(argv): 
  config = FLAGS.config
  spin_shape=(6, 3)
  num_spins = spin_shape[0] * spin_shape[1]
  rng_seq = hk.PRNGSequence(43)
  data_dir = config.data_dir
  output_dir = os.path.join(data_dir, "DMdata/")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  job_id = config.job_id
  h_t_iter = config.h_t_iter
  slice_idx = config.slice_idx
  filename_load = config.filenames_load
  filename_save = config.filenames_save
  
  ens_list = []
  for k, (h, T, i) in enumerate(h_t_iter):
    data = pickle.load(open(data_dir + filename_load[k], "rb"))
    ens_list.append(data)
  ens_dict = dict(zip(h_t_iter, ens_list))  
  
  similarity_fn = functools.partial(diffusion_map.similarity_fn)
  data_sets = []
  for key, params_stacked in ens_dict.items():
    h, T, iteration = key
    data_vars = {}
    params_stacked = utils.slice_along_axis(params_stacked, 0, slice(0, slice_idx), )
    sim_mat = _get_similarity_matrix(similarity_fn, params_stacked)
    ens_idx_array = np.arange(sim_mat.shape[0])
    sim_mat = sim_mat[np.newaxis, np.newaxis, np.newaxis, ...]
    data_vars['S'] = (['h', 'T', 'iter', 'ens_idx_1', 'ens_idx_2'], sim_mat)
    ds = xr.Dataset(data_vars=data_vars, coords=dict(h=[h], T=[T], iter=[iteration], 
      ens_idx_1=ens_idx_array, ens_idx_2=ens_idx_array))
    data_sets.append(ds)

  my_dataset = xr.merge(data_sets)
  my_dataset.to_netcdf(path=output_dir+filename_save)
  with open(output_dir+f'config_{job_id}.json', 'w') as f:
    json.dump(config.to_json(), f)

if __name__ == '__main__':
  app.run(main)