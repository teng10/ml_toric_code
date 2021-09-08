#@ training utilities
import jax
import functools
import jax.numpy as jnp
from jax import grad

def energy_local_fn(config, psi_config, ham, psi, model_params):
  """
  Return E_c
  """
  # return ham.apply(config, psi, model_params)       #E_c = <c|H|psi> 
  return ham.apply(config, psi, model_params) / psi_config      #E_c = <c|H|psi> / psi(c)

def grad_psi(config, psi_config, psi, model_params):
  """
  Return grad_psi
  """
  psi_grad = grad(psi)(model_params, config)
  # print(psi_grad)
  return jax.tree_map(lambda x: x / psi_config, psi_grad)       #grad_psi = grad psi(c) / psi(c)

def energy_grad_psi(config, psi_config, ham, psi, model_params):
  """
  Return E_c * grad_psi
  """
  energy_local = energy_local_fn(config, psi_config, ham, psi, model_params)      #local energy E_c
  psi_grad = grad_psi(config, psi_config, psi, model_params)        #grad_psi
  return jax.tree_map(lambda x: energy_local * x, psi_grad)       

def grad_energy(config, psi_config, psi, model_params, ham):
  """
  Return (E_c * grad_psi, E_c, grad_psi) for config
  """
  return energy_grad_psi(config, psi_config, ham, psi, model_params), energy_local_fn(config, psi_config, ham, psi, model_params), grad_psi(config, psi_config, psi, model_params)

def grad_energy_expectation_fn(batched_configs, batched_psi, psi, model_params, ham):
  """
  Return grad_<E>, <E>, <grad_psi>
  where grad_<E> = <E_c * grad_psi> - <E_c> * <grad_psi>
  """
  compute_energy_grad_fn = functools.partial(grad_energy, psi=psi, model_params=model_params,  ham=ham)
  compute_energy_grad_fn = jax.vmap(compute_energy_grad_fn, in_axes=(0, 0))

  energy_grad_psi_batch, energy_local_batch, grad_psi_batch = compute_energy_grad_fn(batched_configs, batched_psi)
  energy_grad_psi_mean = jax.tree_map(lambda x: jnp.mean(x, axis=0), energy_grad_psi_batch) 
  grad_psi_mean = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_psi_batch) 
  energy_mean = jnp.mean(energy_local_batch, axis=0)
  return jax.tree_multimap(lambda x, y: x - energy_mean * y, energy_grad_psi_mean, grad_psi_mean), jnp.asarray([energy_mean]), grad_psi_mean     

def grad_energy_expectation_gradbatch_fn(batched_configs, batched_psi, psi, model_params, ham):
  """
  Return grad_<E>, <E>, <grad_psi>, grad_psi_batch
  where grad_<E> = <E_c * grad_psi> - <E_c> * <grad_psi>
  """
  compute_energy_grad_fn = functools.partial(grad_energy, psi=psi, model_params=model_params,  ham=ham)
  compute_energy_grad_fn = jax.vmap(compute_energy_grad_fn, in_axes=(0, 0))

  energy_grad_psi_batch, energy_local_batch, grad_psi_batch = compute_energy_grad_fn(batched_configs, batched_psi)
  energy_grad_psi_mean = jax.tree_map(lambda x: jnp.mean(x, axis=0), energy_grad_psi_batch) 
  grad_psi_mean = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_psi_batch) 
  energy_mean = jnp.mean(energy_local_batch, axis=0)
  return jax.tree_multimap(lambda x, y: x - energy_mean * y, energy_grad_psi_mean, grad_psi_mean), jnp.asarray([energy_mean]), grad_psi_mean, grad_psi_batch   
