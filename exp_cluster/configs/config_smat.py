import ml_collections
import itertools
from ml_collections import config_dict
import os

def get_config():
  config = ml_collections.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  #
  config.iter_list = [2]
  h_params = [0.3, 0.4, 0.5, 0.6]
  t_params = [0.1, 0.3, 0.7]
  # h_and_t = list(itertools.product(h_params, t_params))
  config.h_t_iter = list(itertools.product(h_params, t_params, config.iter_list))
  config.slice_idx = 6000

  config.data_dir = "/n/home11/yteng/experiments/optimization/data/5978142/ensemble/7592828/"
  # config.output_dir = os.path.join(config.data_dir, "estimates/")
  config.filenames_load = [f"samples_(6, 3)_hz{h}_T{T}_iter{i}.p" for (h, T, i) in config.h_t_iter]
  config.filenames_save = f"S_mat.nc"
  
  return config

