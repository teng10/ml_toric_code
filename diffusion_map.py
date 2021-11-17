#@title Diffusion map 
import jax
import jax.numpy as jnp
import utils

def extract_V_F_params(param_dict):
  if len(list(param_dict.keys())) > 1:
    raise ValueError("Dictionary has more than one key")
  for key in param_dict:
    bF = param_dict[key]['bF']
    wF = param_dict[key]['wF']
    bV = param_dict[key]['bV']
    wV = param_dict[key]['wV']    
  F_array = jnp.concatenate((bF, wF))
  V_array = jnp.concatenate((bV, wV))
  return F_array, V_array

def similarity_fn(w1, w2):
	"""
	Takes param dictionaries w1, w2, return similarity
	"""
	def _S_max(array1, array2):
		w_difference_cos = jnp.array([jnp.cos(2. * (array1 - array2)), jnp.cos(2. * (-array1 - array2))])
		w_cos_mean = jnp.mean(w_difference_cos, axis=1)
		w_cos_max = jnp.max(w_cos_mean, axis=0)
		return jnp.mean(w_cos_max)
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = _S_max(F_array_1, F_array_2)
	V_max = _S_max(V_array_1, V_array_2)
	return (1. + (F_max + V_max) / 2.) /2. 

def kernel_fn(similarity, epsilon):
	return jnp.exp(-(1. - similarity) / (2. * epsilon))

# def kernel_fn(w1, w2, epsilon):
# 	return jnp.exp(-(1. - similarity_fn(w1, w2)) / (2. * epsilon))

# def kernel_mat(all_w, epsilon):
# 	stacked_w_dict = utils.stack_along_axis(all_w, 0)
# 	kernel_vec = jax.vmap(kernel_fn, in_axes=(0, None, None))
# 	kernel_vec_vec = jax.vmap(kernel_vec, in_axes=(None, 0, None))
# 	return kernel_vec_vec(stacked_w_dict, stacked_w_dict, epsilon)

# def transition_mat(all_w, epsilon):
# 	"""Return both kernal matrix and a similar matrix A (which is symmetric)"""
# 	K_mat = kernel_mat(all_w, epsilon)
# 	z1 = jnp.sum(K_mat, axis=1)
# 	z2 = jnp.sum(K_mat, axis=0)
# 	A_mat = K_mat / jnp.sqrt(jnp.outer(z1, z2))
# 	return K_mat, A_mat
	
# def kernel_mat(all_w, epsilon):
# 	kernel_vec = jax.vmap(kernel_fn, in_axes=(0, None, None))
# 	kernel_vec_vec = jax.vmap(kernel_vec, in_axes=(None, 0, None))
# 	return kernel_vec_vec(all_w, all_w, epsilon)

# def transition_mat(all_w, epsilon):
# 	"""Return both kernal matrix and a similar matrix A (which is symmetric)"""
# 	K_mat = kernel_mat(all_w, epsilon)
# 	z1 = jnp.sum(K_mat, axis=1)
# 	z2 = jnp.sum(K_mat, axis=0)
# 	A_mat = K_mat / jnp.sqrt(jnp.outer(z1, z2))
# 	return K_mat, A_mat

def transition_mat(K_mat, epsilon, return_z=False):
	"""Return both kernal matrix and a similar matrix A (which is symmetric)"""
	# K_mat = kernel_mat(all_w, epsilon)
	z1 = jnp.sum(K_mat, axis=1)
	z2 = jnp.sum(K_mat, axis=0)
	A_mat = K_mat / jnp.sqrt(jnp.outer(z1, z2))
	if return_z:
		return A_mat, jnp.diag(1. / jnp.sqrt(z2))
	return A_mat
