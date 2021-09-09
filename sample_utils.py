#@title Utility functions
import jax
import jax.numpy as jnp
import numpy as np

def init_samples(rng, n_sites, batch_size=1):
  cs = jax.random.randint(rng, shape=(batch_size, n_sites), minval=0, maxval=2)
  cs = (cs * 2 - 1).astype(float)
  if batch_size == 1:
    return jnp.squeeze(cs, axis=0)
  return cs

#@title Define Potential Sampling Methods
def vertex_bond_sample(config, bond):
  "Act with pauliX/PauliY operator"
  new_config = jax.ops.index_update(config, bond, config[bond] * (-1))
  return new_config

def face_bond_sample(config, bond):
  "Act with pauliZ operator"
  new_config = config       #sigma_z does not change the spins
  return new_config

def _batch_iterator(iterator, batch_size=1):
  """
  Return iterator with size=batch_size.
  For computing exact overlap to iterate over all configurations. 
  """
  cs = []
  count = 0
  for c in iterator:
    cs.append(c)
    count += 1
    if count == batch_size:
      cs_out = np.stack(cs)
      cs = []
      count = 0
      yield cs_out
  if not cs:
    raise StopIteration()
  yield np.stack(cs)