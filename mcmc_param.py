import jax
import jax.numpy as jnp
import jax.experimental.host_callback as jcb
import numpy as np
import utils
import functools
import tc_utils

# _make_move_param_fn = lambda param, noise: jax.tree_map(lambda x, y: x + y, param, noise)

# def propose_param_fn(key, param_dict, p_mpar, amp_noise):
# 	"""Propose with probabilty p_mpar m particle excitations; otherwise random noise. 

# 	Returns:
# 	array representing a packed `tree` and `unpack` function.
# 	"""	
# 	key_flip, key_propose = jax.random.split(key, 2)
# 	random_num = jax.random.uniform(key_flip)
# 	condition = random_num < p_mpar
# 	return jax.lax.cond(condition, lambda _: tc_utils.generate_m_particles_param(key_propose, param_dict), 
# 		lambda _: tc_utils.generate_uniform_noise_param(key_propose, param_dict, amp_noise), None)
	# return jnp.where(condition, tc_utils.generate_m_particles_param(key_propose, param_dict),
	# 	tc_utils.generate_uniform_noise_param(key_propose, param_dict, amp_noise))

def _accept_E_fn(key, energy_param, energy_propose, T, return_diff=False):
	"""Return condition of acceptance."""
	energy_diff = - (energy_propose - energy_param) / T
	random_num = jax.random.uniform(key, shape=energy_diff.shape)
	random_num_log = jnp.log(random_num)
	if return_diff:
		return energy_diff > random_num_log, energy_diff
	return energy_diff > random_num_log

def _update_param(
	key, 
	param, param_propse, energy_param, 
	accept_fn, 
	estimate_ET_fn):
	"""Note that energy already implicitly has T."""
	key_accept, key_E = jax.random.split(key, 2)
	# Modify estimate_ET_fn to be compatible with non-fitted computation
	# estimate_ET_fn_part = functools.partial(estimate_ET_fn, key=key_E)
	energy_propose, *energy_others = estimate_ET_fn(key_E, param_propse)
	# energy_propose, *energy_others = estimate_ET_fn(param_propse)
	# energy_propose = jcb.call(estimate_ET_fn_part, param_propse, 
	# 	result_shape=jax.ShapeDtypeStruct((), np.float32))
	condition = accept_fn(key_accept, energy_param, energy_propose)
	# param_new = jnp.where(condition, param_propse, param)
	# energy_new = jnp.where(condition, energy_propose, energy_param)
	param_new = jax.lax.cond(condition, lambda _: param_propse, lambda _: param, None)
	# jax.tree_map(lambda leaf_propose, leaf: jnp.where(condition, leaf_propose, leaf), param_propse, param)
	energy_new = jnp.where(condition, energy_propose, energy_param)	
	return param_new, energy_new, condition


def _update_param_chain(key, param, energy_param, 
						len_chain, 	
						estimate_ET_fn, 
						accept_fn,
						propose_move_param_fn,):
	def _mcmc_walk(carry, inputs):
		param, energy_param, num_accepts = carry
		key_propose, key_accept = inputs
		param_propse = propose_move_param_fn(key_propose, param)
		param_new, energy_new, condition = _update_param(key_accept, param, param_propse, energy_param, 
														accept_fn, estimate_ET_fn)
		num_accepts = num_accepts + condition
		return (param_new, energy_new, num_accepts), (param_new, energy_new)

	# rngs = tuple(utils.split_key(key, (2, len_chain, 2)))	
	rngs = utils.split_key(key, (2 * len_chain, 2))
	rngs = jnp.split(rngs, 2, axis=0)
	return jax.lax.scan(f=_mcmc_walk, init=(param, energy_param, 0), xs=rngs)

def energy_sampling_mcmc(key, init_params, 
						len_chain_burn, len_chain, 
						estimate_ET_fn, 
						propose_move_param_fn, 
						T, 						
						accept_fn=_accept_E_fn, 
						return_results=False):
	num_chains = len(init_params)
	key_param, key_E_init = jax.random.split(key, 2)
	rngs_E = utils.split_key(key_E_init, (num_chains, 2))
	init_params_stacked = utils.stack_along_axis(init_params, 0)
	estimate_ET_vec = jax.vmap(estimate_ET_fn, in_axes=(0, 0))
	# Modify for non-gitted computation
	# estimate_ET_vec_part = functools.partial(estimate_ET_vec, key=rngs_E)
	init_energies, *init_energies_others = estimate_ET_vec(rngs_E, init_params_stacked)
	# init_energies = jcb.call(estimate_ET_vec_part, init_params_stacked, 
	# 	result_shape=jax.ShapeDtypeStruct((num_chains,), np.float32))
	# print(init_energies.shape)
	# print(jax.tree_structure(init_params_stacked))
	accept_fn_T = functools.partial(accept_fn, T=T)
	_uupdate_param_chain_fn = functools.partial(_update_param_chain, estimate_ET_fn=estimate_ET_fn, 
												accept_fn=accept_fn_T, 
												propose_move_param_fn=propose_move_param_fn)
	_update_param_chain_vec = jax.vmap(_uupdate_param_chain_fn, in_axes=(0, 0, 0, None))
	key_burn, key_sample = jax.random.split(key_param, 2)
	rngs_burn = utils.split_key(key_burn, (num_chains, 2))
	if len_chain_burn is not None:
		(params_burn, energies_burn, _), _ = _update_param_chain_vec(rngs_burn, init_params_stacked, init_energies, len_chain_burn)
	else:
		params_burn, energies_burn = (init_params_stacked, init_energies)
	# print(jax.tree_structure(params_burn))
	# # params_burn_stacked = utils.stack_along_axis(params_burn, 0)
	# print(jax.tree_structure(params_burn))
	rngs_sample = utils.split_key(key_sample, (num_chains, 2))
	(param_new, energy_new, num_accepts), (params_sample, energies_sample) = _update_param_chain_vec(rngs_sample, params_burn, energies_burn, len_chain)
	# if return_results:
	# 	return (param_new, energy_new, num_accepts/len_chain), (params_sample, energies_sample)
	return num_accepts/len_chain, params_sample, energies_sample








