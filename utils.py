#@ utils
import jax
import jax.numpy as jnp
import math
import numpy as np

# Some utilities borrowed from jax.cfd codebase. 

round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
round_to_2 = lambda x: round_to_n(x, 2)

def split_key(key, new_shape):
  new_shape = np.array(new_shape)
  # keys_dim = new_shape[:-1]
  # print(keys_dim)
  rng = jax.random.split(key, np.prod(new_shape)//new_shape[-1])      #Split the keys based on batch size and steps
  rngs = jnp.reshape(rng, new_shape)
  return rngs
  
def get_pack_unpack_fns(pytree):
  """Packs `tree` to a flattened vector.

  Args:
    pytree: pytree to be packed.

  Returns:
    array representing a packed `tree` and `unpack` function.
  """
  flat, treedef = jax.tree_flatten(pytree)
  shapes = [f.shape for f in flat]
  flat = [f.ravel() for f in flat]
  splits = np.cumsum(np.array([f.shape[0] for f in flat]))[:-1]

  def pack_fn(pytree):
    flat, _ = jax.tree_flatten(pytree)
    flat = [f.ravel() for f in flat]
    packed = jnp.concatenate(flat)
    return packed

  def unpack_fn(array):
    split = jnp.split(array, splits, 0)
    split = [s.reshape(new_shape) for s, new_shape in zip(split, shapes)]
    return jax.tree_unflatten(treedef, split)
  return pack_fn, unpack_fn

def concat_along_axis(pytrees, axis):
  """Concatenates `pytrees` along `axis`."""
  concat_leaves_fn = lambda *args: jnp.concatenate(args, axis)
  return jax.tree_map(concat_leaves_fn, *pytrees)

def stack_along_axis(pytrees, axis):
  """Concatenates `pytrees` along `axis`."""
  concat_leaves_fn = lambda *args: jnp.stack(args, axis)
  return jax.tree_map(concat_leaves_fn, *pytrees)