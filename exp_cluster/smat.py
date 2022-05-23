#@title Estimating operators in a dictionary

import re
import datetime
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import functools
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
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
FLAGS = flags.FLAGS


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
  #Compute similarity
  similarity_fn = functools.partial(diffusion_map.similarity_fn_np)
  data_vars = {}
  sim_mat = diffusion_map._get_similarity_matrix_np(similarity_fn, params_loaded)
  sim_mat = sim_mat[np.newaxis, np.newaxis, np.newaxis, ...]
  data_vars['S'] = (['h', 'T', 'iter', 'ens_idx_1', 'ens_idx_2'], sim_mat)
  ds = xr.Dataset(data_vars=data_vars, coords=dict(h=[h], T=[T], iter=[iteration], 
    ens_idx_1=ens_idx_array, ens_idx_2=ens_idx_array))
  # save dataset
  ds.to_netcdf(path=output_dir+filename_save)
  # Save config file as json
  with open(output_dir+f'config_{job_id}.json', 'w') as f:
    json.dump(config.to_json(), f)

if __name__ == '__main__':
  app.run(main)