import ml_collections
import itertools
from ml_collections import config_dict

def get_config():
  config = ml_collections.ConfigDict()
  # job properties
  config.job_id = config_dict.placeholder(int)
  config.file_id = config_dict.placeholder(int)

  config.iter = 2
  config.spin_shape = (6, 3)
  config.sector_list = [1, 2, 3, 4]
  config.angle = 0. #Angle of the field wrt z-axis
  config.noise_amp = 0.2 #Amount of noise added for each h
  config.p_mparticle = 0.3 #Probability of doing sign flip operation that creates m particle in Kitaev state
  # MCMC parameters
  config.burn_E_factor = 500
  config.num_samples_E = 200
  config.len_chain_E = 30
  # number of ensembles to generate for each sector
  config.num_samples = 3
  config.len_chain= 150
  # T and h parameters to generate
  # h_params = [0.]
  h_params = [0., 0.1, 0.2, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 1.]
  t_params = [0.1, 0.3, 0.7]
  config.h_and_t = list(itertools.product(h_params, t_params))

  config.filenames_ens = [f"samples_{config.spin_shape}_hz{h}_T{T}_iter{config.iter}.p" for (h, T) in config.h_and_t]
  config.filenames_ens_property = [f"energies_accept_{config.spin_shape}_hz{h}_T{T}_iter{config.iter}.nc" for (h, T) in config.h_and_t]

  config.data_dir = "/Volumes/GoogleDrive/My Drive/Projects/ML_toric_code/Data_cluster/data/10331255/"  
  return config


