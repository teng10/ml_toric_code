import ml_collections
import itertools

def get_config():
  config = ml_collections.ConfigDict()
  config.iter = 2
  config.field2 = 'tom'
  config.burn_E_factor = 500
  config.num_samples_E = 200
  config.len_chain_E = 30
  config.len_chain= 150
  config.tuple = (1, 2, 3)
  #h_params = [0.35, 0.45, 0.55, 0.65]
  h_params = [0., 0.1, 0.2, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 1.]
  t_params = [0.1, 0.3, 0.7]
  config.h_and_t = list(itertools.product(h_params, t_params))

  return config

