import ml_collections
import itertools
from ml_collections import config_dict

def get_config():
  config = ml_collections.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.file_id = config_dict.placeholder(int)
  #

  config.iter_list = [2]
  t_params = [0.1, 0.3, 0.7]
  h_params = [0., 0.1, 0.2, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 1.]
  # t_params = [0.1]
  # h_params = [0.]
  config.h_t_iter = list(itertools.product(h_params, t_params, config.iter_list))

  config.data_dir = "/Volumes/GoogleDrive/My Drive/Projects/ML_toric_code/Data_cluster/data/5978142/ensemble/9952773/"
  
  config.filenames_load = [f"samples_(6, 3)_hz{h}_T{T}_iter{i}.p" for (h, T, i) in config.h_t_iter]
  config.filenames_save = [f"overlap_mat_(6, 3)_hz{h}_T{T}_iter{i}.nc" for (h, T, i) in config.h_t_iter]
  
  return config
