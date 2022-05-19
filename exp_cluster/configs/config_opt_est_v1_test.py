import ml_collections
import itertools
from ml_collections import config_dict

def get_config():
  config = ml_collections.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.file_id = config_dict.placeholder(int)
  #
  config.est_desc = "WL_H_S_O"
  config.compute_vector = True
  #
  config.sector_list = [1, 2]
  config.iter_list = [0, 1, 2]
  config.model_name = 'rbm'
  config.burn_E_factor = 5
  config.num_samples_E = 2
  config.len_chain_E = 3
  config.spin_shape = (6, 3)
  config.angle = 0. #Angle of the field wrt z-axis
  config.noise_amp = 0.2 #Amount of noise added for each h
  # config.h_params = [0., 0.1, 0.2, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 1.]
  config.h_params = [0., 0.325]

  config.data_dir = "/Volumes/GoogleDrive/My Drive/Projects/ML_toric_code/Data_cluster/data/10331255/"
  config.filename_ens = "params_dict.p"
  config.filename_vec = [f"vec_{i}.nc" for i in config.iter_list]
  config.filenames_ens_ppt = [f"{config.est_desc}_{config.spin_shape}_{i}.nc" for i in config.iter_list]
  
  return config

