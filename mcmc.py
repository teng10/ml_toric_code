#@title MCMC module
import jax
import jax.numpy as jnp



def _update_config(key, config, config_propose, psi_c, psi, model_params):
  """
  Takes config and proposed config and outputs the updated configuration based on diffusion rule. 
  P(accept c') = min(1, |psi(c')|^2/|psi(c)|^2), but work with log probability.
  """
  psi_propose = psi(model_params, config_propose)
  log_ratio = 2 * (jnp.log(jnp.abs(psi_propose)) - jnp.log(jnp.abs(psi_c)))    #Compute lof of |psi|^2 ratio
  random_num = jax.random.uniform(key, shape = log_ratio.shape, minval=0, maxval=1)       #Generate a random float between 0 and 1
  random_num_log = jnp.log(random_num)        #Take log of random float
  condition = log_ratio > random_num_log      #Get probability of log_ratio for True condition
  new_config = jnp.where(condition, config_propose, config)      #Choose between config and proposed config
  new_psi = jnp.where(condition, psi_propose, psi_c)
  return new_config, new_psi, condition

def propose_move_fn(key, config, p, vertex_bonds):
  """
  Propose spin flips with probability 'p'; vertex flips with probability '1-p'
  """
  def _propse_spin_flips(key, config):
    num_spin = config.shape[0]
    spin = jax.random.randint(key, shape=(1,), minval=0, maxval=num_spin)
    return spin
  
  def _propse_vertex_flips(key, config):
    num_bonds = vertex_bonds.shape[0]
    vertex = jax.random.randint(key, (1,), minval=0, maxval=num_bonds)
    # return vertex_bonds[tuple(vertex)]
    return vertex_bonds[vertex]

  new_key, sub_key = jax.random.split(key, num=2)
  # random_num = jax.random.uniform(sub_key, shape=(1,))
  random_num = jax.random.uniform(sub_key)
  condition = random_num < p
  proposed_move = jnp.where(condition, _propse_spin_flips(new_key, config), 
                            _propse_vertex_flips(new_key, config))

  return proposed_move

def update_chain(key, config, psi_c, model_params, len_chain, psi,  propose_move_fn, make_move_fn, p):
  """
  For a given chain with initial 'config', attempts to walk 'len_chain' steps and return 'new_config'. 
  p: probability of propsing spin flips
  psi: model.apply
  """
  def _mcmc_fc(carry, inputs):
    config, psi_c, num_updates  = carry
    rngs = inputs
    rng1, rng2 = rngs       #Define the splitted keys for making moves and update config

    move = propose_move_fn(rng1, config, p)     #Propose a move
    config_propose = make_move_fn(config, move)       #Make the move
    new_config, new_psi, condition = _update_config(rng2, config, 
                                                   config_propose, psi_c, psi, model_params) #Update config
    
    num_updates = num_updates + condition      #Keep track of number of accepted new configs
    return (new_config, new_psi, num_updates), None

  rng = jax.random.split(key, num=2 * len_chain)
  rngs = jnp.split(rng, 2, axis=0)
  
  (new_config, new_psi, num_updates), _ = jax.lax.scan(f=_mcmc_fc, init=(config, psi_c, jnp.array(0)), xs=(rngs))
  return new_config, new_psi, num_updates
