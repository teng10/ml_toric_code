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
import os
import sys
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
import ml_collections
from ml_collections.config_flags import config_flags

config_flags.DEFINE_config_file('config')
FLAGS = flags.FLAGS
DataKey = collections.namedtuple('DataKey', ['sector', 'h_field', 'iteration'])

def _get_op_dict(directions, loop_indices, spin_shape, h, angle):
  # Build operator dictionary
  num_spins = spin_shape[0] * spin_shape[1]
  op_dict = {}
  ham = tc_utils.set_up_ham_field_rotated(spin_shape, h, angle) # Build hamiltonian operator
  op_dict['ham'] = ham
  pauliZ =  operators.ToricCodeHamiltonianRotated(Jv=0., Jf=0., h=1., hx=0., face_bonds=[], vertex_bonds=[], pauli_bonds=np.arange(0, num_spins, 1))  
  op_dict['pauliZ'] = pauliZ
  for dir in directions:  # Build sets of wilson loop operators
    for idx in loop_indices:
      label = [f'WLX{idx}', f'WLY{idx}']
      loop_x = operators.WilsonLXBond(bond=bonds.wilson_loops(spin_shape, dir, idx))
      op_dict[label[dir]] = loop_x
  return op_dict

def main(argv): 
  config = FLAGS.config
  job_id = config.job_id
  file_id = config.file_id
  est_desc = config.est_desc
  h_field_array = np.array(config.h_params)
  rng_seq = hk.PRNGSequence(42 + file_id)
  data_dir = config.data_dir
  output_dir = config.output_dir
  spin_shape = config.spin_shape
  sector_list = config.sector_list
  iteration_list = config.iter_list
  iteration = iteration_list[file_id]
  # model and sampling property
  angle = config.angle
  noise_amp = config.noise_amp
  # Parameters for mcmc energy
  len_chain_E = config.len_chain_E
  burn_E_factor = config.burn_E_factor
  num_samples_E = config.num_samples_E
  filename_ens = config.filename_ens
  filename_ens_ppt = config.filenames_ens_ppt[file_id]
  filename_vec = config.filename_vec[file_id]

  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if not os.path.exists(data_dir + filename_ens):  
    full_data_tuple = tc_utils.load_opt_data_tuple_dicts(sector_list, iteration_list, data_dir, DataKey)
    params_dict = full_data_tuple[0]
    pickle.dump(params_dict, open(data_dir + filename_ens, 'wb'))  
  else:
    params_dict = pickle.load(open(data_dir + filename_ens, 'rb'))
      
  # Parameters for mcmc energy
  num_spins = spin_shape[0] * spin_shape[1]
  len_chain_burn_E = burn_E_factor * num_spins
  directions = [0, 1]   # directions of loop to compute
  batch_size = 64   # batch size for exact computation of vector
  loop_indices = range(spin_shape[1])
  # estimates properties of wv associated to each parameter
  results_desc = ["MCMC_ev", "MCMC_std", "MCMC_local_vals", "MCMC_accept_mean"]
  full_data = []
  for sec in sector_list:
    sec_data = []
    for h in tqdm(h_field_array, desc="h, mcmc estimates"):
      h_data = []
      for i in [iteration, ]:
        param = params_dict[(sec, h, i)]
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

  # compute vectors explicitly
  if config.compute_vector:
    vec_list_h_sec = []
    for h in tqdm(h_field_array, desc="h"):
      vec_list_sec = []
      for sec in sector_list:
        param = params_dict[(sec, h, iteration)]
        vec = exact_comp.get_vector(num_spins, batch_size, psi_apply, param)
        vec_list_sec.append(vec)
      vec_list_h_sec.append(np.stack(vec_list_sec, 0))
    vec_h_sec  = np.stack(vec_list_h_sec, 0)     
    vec_h_sec = np.expand_dims(vec_h_sec, 0)
    data_vars_vec = {}
    data_vars_vec['psi'] = (['iter', 'h', 'sec', 'c'], vec_h_sec)      
    ds_vec = xr.Dataset(data_vars=data_vars_vec, coords=dict(h=h_field_array, sec=sector_list, iter=[iteration], 
    c=np.arange(2**num_spins)))
    ds_vec.to_netcdf(path=output_dir+filename_vec)

    # compute overlap
    for k, h in enumerate(h_field_array):
      overlaps = []
      for sec in sector_list:
        vecs = vec_h_sec[0, k, :]
        overlaps.append(exact_comp._get_overlap_matrix(vecs))
      data_vars['overlaps_exact'] = (['iter', 'h', 'sec', 'sec2'], np.stack(overlaps, 0)[np.newaxis, ...])

    # compute fidelity
    for k, sec in enumerate(sector_list):
      fidelities = []
      for h in h_field_array:
        vecs = vec_h_sec[0, :, k]
        fidelities.append(exact_comp._get_overlap_matrix(vecs))
      data_vars['fidelities_exact'] = (['iter', 'sec', 'h', 'h2'], np.stack(fidelities, 0)[np.newaxis, ...])    

  my_dataset = xr.Dataset(data_vars=data_vars, coords=dict(sec=sector_list, h=h_field_array, iter=[iteration], 
    sec2=sector_list, h2=h_field_array))

  my_dataset.to_netcdf(path=output_dir+filename_ens_ppt)
  with open(output_dir+f'config_{job_id}.json', 'w') as f:
    json.dump(config.to_json(), f)

if __name__ == '__main__':
  app.run(main)