import ml_collections
import itertools
from ml_collections import config_dict
import os

def get_config():
  config = ml_collections.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.file_id = config_dict.placeholder(int)
  #
  sec_list = [1, 2, 3, 4]
  iter_list = [0, 1, 2]
  config.model_name = 'rbm'
  config.num_steps = 4
  config.num_chains = 5
  config.learning_rate = 0.001
  config.spin_flip_p = 0.3
  config.burn_in_factor = 6
  config.spin_shape = (6, 3)
  config.angle = 0. #Angle of the field wrt z-axis
  config.epsilon = 0.2 #Amount of noise added for each h
  # config.h_params = [0., 0.1, 0.2, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 1.]
  config.h_params = [0., 0.125]
  config.sec_iter = list(itertools.product(sec_list, iter_list))

  config.data_dir = "/Volumes/GoogleDrive/My Drive/Projects/ML_toric_code/Data_cluster/data/test/"
  config.filenames_save = [f"results_{config.spin_shape}_{sec}_{i}.p" for (sec, i) in config.sec_iter]
  
  return config

