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

def main(argv):
  print(f'Program has started with args: {argv}')
  h_field = float(argv[1])
  T = float(argv[2])
  spin_shape=(6, 3)
  num_spins = spin_shape[0] * spin_shape[1]
  angle = 0.
  rng_seq = hk.PRNGSequence(42 + sector * int(argv[3]))
  noise_amp = 0.2
  p_mparticle = 0.3
  model = hk.without_apply_rng(hk.transform(wavefunctions.fwd_noise))
  psi_apply = functools.partial(model.apply, spin_shape=spin_shape)
  # Parameters for mcmc energy
  len_chain_E=30
  burn_E_factor = 5
  num_samples_E = 2
  # Parameters for mcmc_param
  len_chain= 3
  burn_factor= None
  num_samples=2
  # sector_labels = np.concatenate(([np.full((num_samples * len_chain,), i) for i in sector_list]))     # Create labels for each batch of samples 
  file_path = argv[4]
  # jit function
  generate_samples_T_fn = functools.partial(
      notebook_fn.generate_samples_T, spin_shape=spin_shape, psi_apply=psi_apply,
      len_chain_E=len_chain_E, burn_E_factor=burn_E_factor, num_samples_E=num_samples_E, 
      len_chain=len_chain, burn_factor=burn_factor, num_samples=num_samples, 
      noise_amp=noise_amp, p_flip=p_mparticle)
  generate_samples_T_jit = jax.jit(generate_samples_T_fn, static_argnums=(8, ))

  new_samples_list = []
  new_energies_list = []
  accept_rate_list = []
  screened_params_dict = pickle.load(open(file_path+"params_screened_temp.p", "rb"))
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

  pickle.dump(samples_all_secs, open(file_path+"/ensemble/"+file_name_samples, 'wb'))
  pickle.dump(energies_all_secs, open(file_path+"/ensemble/"+file_name_energies, 'wb'))
  pickle.dump(accepts_all_secs, open(file_path+"/ensemble/"+file_name_accept_rate, 'wb'))

if __name__ == '__main__':
  app.run(main)