#@ evaluation utilities
import jax
import functools
import jax.numpy as jnp
from jax import grad
import numpy as np

def op_local_fn(config, psi_config, op, psi, model_params):
  """
  Return O_c
  """
  # return ham.apply(config, psi, model_params)       #O_c = <c|O|psi> 
  return op.apply(config, psi, model_params) / psi_config      #O_c = <c|O|psi> / psi(c) 

def op_expectation_fn(batched_configs, batched_psi, psi, model_params, op):
  """
  Return <O>
  """
  compute_op_fn = functools.partial(op_local_fn, psi=psi, model_params=model_params,  op=op)
  compute_op_fn_vec = jax.vmap(compute_op_fn, in_axes=(0, 0))

  op_local_batch = compute_op_fn_vec(batched_configs, batched_psi)  
  return jnp.mean(op_local_batch), jnp.std(op_local_batch, axis=0) / np.sqrt(op_local_batch.shape[0]), op_local_batch