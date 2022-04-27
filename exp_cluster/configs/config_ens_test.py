import ml_collections
import itertools

def get_config():
  config = ml_collections.ConfigDict()
  config.iter = 2
  config.field2 = 'tom'
  config.burn_E_factor = 5
  config.num_samples_E = 2
  config.len_chain_E = 3
  config.len_chain= 3
  config.tuple = (1, 2, 3)
  h_params = [0., 1.]
  t_params = [0.1, 0.3, 0.7]
  config.h_and_t = list(itertools.product(h_params, t_params))
  
  return config

