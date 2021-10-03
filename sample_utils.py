#@title Utility functions
import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, Callable, Tuple, Union
PyTree = Any

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

def convert_dict_array_to_list(dicts_list):
  batch = jax.tree_leaves(dicts_list[0])[0].shape[0]
  array_to_list_fn = lambda tree: [jax.tree_map(lambda x: x[i, ...], tree) for i in range(batch)]
  jax.tree_map(array_to_list_fn, *dicts_list)

def split_axis(
    inputs: PyTree,
    axis: int,
    keep_dims: bool = False
) -> Tuple[PyTree, ...]:
  """Splits the arrays in `inputs` along `axis`.
  Args:
    inputs: pytree to be split.
    axis: axis along which to split the `inputs`.
    keep_dims: whether to keep `axis` dimension.
  Returns:
    Tuple of pytrees that correspond to slices of `inputs` along `axis`. The
    `axis` dimension is removed if `squeeze is set to True.
  Raises:
    ValueError: if arrays in `inputs` don't have unique size along `axis`.
  """
  arrays, tree_def = jax.tree_flatten(inputs)
  axis_shapes = set(a.shape[axis] for a in arrays)
  if len(axis_shapes) != 1:
    raise ValueError(f'Arrays must have equal sized axis but got {axis_shapes}')
  axis_shape, = axis_shapes
  splits = [jnp.split(a, axis_shape, axis=axis) for a in arrays]
  if not keep_dims:
    splits = jax.tree_map(lambda a: jnp.squeeze(a, axis), splits)
  splits = zip(*splits)
  return tuple(jax.tree_unflatten(tree_def, leaves) for leaves in splits)




