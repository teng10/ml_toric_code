#@ utils
import jax
import jax.numpy as jnp
import math
import numpy as np
import functools
from typing import Any, Callable, Tuple, Union
PyTree = Any

# Some utilities borrowed from jax.cfd codebase. 

round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
round_to_2 = lambda x: round_to_n(x, 2)

def _normalize_axis(axis: int, ndim: int) -> int:
  """Validates and returns positive `axis` value."""
  if not -ndim <= axis < ndim:
    raise ValueError(f'invalid axis {axis} for ndim {ndim}')
  if axis < 0:
    axis += ndim
  return axis

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
    
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
  splits = tuple(np.cumsum(np.array([f.shape[0] for f in flat]))[:-1])

  def pack_fn(pytree):
    flat, _ = jax.tree_flatten(pytree)
    flat = [f.ravel() for f in flat]
    packed = jnp.concatenate(flat)
    return packed

  def unpack_fn(array):
    split = jnp.split(array, splits, 0)
    # split = jnp.split(array, tuple(splits), 0)
    split = [s.reshape(new_shape) for s, new_shape in zip(split, shapes)]
    return jax.tree_unflatten(treedef, split)
  return pack_fn, unpack_fn

def concat_along_axis(pytrees, axis):
  """Concatenates `pytrees` along `axis`."""
  concat_leaves_fn = lambda *args: jnp.concatenate(args, axis)
  return jax.tree_map(concat_leaves_fn, *pytrees)

def stack_along_axis(pytrees, axis):
  """Concatenates `pytrees` along `axis`."""
  stack_leaves_fn = lambda *args: jnp.stack(args, axis)
  return jax.tree_map(stack_leaves_fn, *pytrees)

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

def slice_along_axis(
    inputs: PyTree,
    axis: int,
    idx: Union[slice, int],
    expect_same_dims: bool = True
) -> PyTree:
  """Returns slice of `inputs` defined by `idx` along axis `axis`.
  Args:
    inputs: array or a tuple of arrays to slice.
    axis: axis along which to slice the `inputs`.
    idx: index or slice along axis `axis` that is returned.
    expect_same_dims: whether all arrays should have same number of dimensions.
  Returns:
    Slice of `inputs` defined by `idx` along axis `axis`.
  """
  arrays, tree_def = jax.tree_flatten(inputs)
  ndims = set(a.ndim for a in arrays)
  if expect_same_dims and len(ndims) != 1:
    raise ValueError('arrays in `inputs` expected to have same ndims, but have '
                     f'{ndims}. To allow this, pass expect_same_dims=False')
  sliced = []
  for array in arrays:
    ndim = array.ndim
    slc = tuple(idx if j == _normalize_axis(axis, ndim) else slice(None)
                for j in range(ndim))
    sliced.append(array[slc])
  return jax.tree_unflatten(tree_def, sliced)


def shape_structure(pytree):
  return jax.tree_map(lambda x: x.shape, pytree)

def iterate_batch(fn, samples, axis, batch=30):
  # we have contract that all leafs of samples have the same `axis` dim;
  # we can write a check that this is true and extract that value;
  # if that value is less or eq than batch, just return fn(samples)
  # otherwise use your code;
  dim_axis = jax.tree_leaves(shape_structure(samples))[axis]
  if dim_axis < batch:
    return fn(*samples)
  sections = dim_axis // batch 
  # print(sections)
  split_fn = functools.partial(jnp.array_split, indices_or_sections=sections, axis=axis)
  batched_inputs = jax.tree_map(lambda x: jnp.stack(split_fn(x), axis=0), samples)
  # print(shape_structure(batched_inputs))
  batched_inputs_list = split_axis(batched_inputs, axis=0)
  # print(shape_structure(batched_inputs_list))
  # print(batched_inputs)
  batched_results = []
  for inputs in batched_inputs_list:
    results = fn(*inputs)
    batched_results.append(results)
  return concat_along_axis(batched_results, axis=0)
