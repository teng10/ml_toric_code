#@title Diffusion map 
import jax
import jax.numpy as jnp
import numpy as np
import utils
import itertools
import einops
import tc_utils

def _get_similarity_matrix_np(similarity_fn, params_stacked):
  # params_list = utils.split_axis(params_stacked, axis=0)
  num_params = jax.tree_leaves(utils.shape_structure(params_stacked))[0]
  s_mat = np.zeros((num_params, num_params))
  for i in range(num_params):
    for j in range(i, num_params):
      param1 = utils.slice_along_axis(params_stacked, 0, i)
      param2 = utils.slice_along_axis(params_stacked, 0, j)
      sim = similarity_fn(param1, param2)
      s_mat[i, j] = sim
      s_mat[j, i] = sim
  return s_mat
  
def _get_similarity_matrix(similarity_fn, params_stacked):
  similarity_vec = jax.vmap(jax.vmap(similarity_fn, (0, None)), (None, 0))
  return similarity_vec(params_stacked, params_stacked)
  
# def extract_V_F_params(param_dict):
# 	keys = list(param_dict.keys())
# 	if len(keys) > 1:
# 		raise ValueError("Dictionary has more than one key")
# 	key = keys[0]
# 	bF = param_dict[key]['bF']
# 	wF = param_dict[key]['wF']
# 	bV = param_dict[key]['bV']
# 	wV = param_dict[key]['wV']
# 	F_array = jnp.concatenate((bF, wF))
# 	V_array = jnp.concatenate((bV, wV))
# 	return F_array, V_array

def extract_V_F_params(param_dict, np_only=False):
	keys = list(param_dict.keys())
	if len(keys) == 1:
		key = keys[0]
		if key == 'rbm' or key == 'rbm_noise':
			bF = param_dict[key]['bF']
			wF = param_dict[key]['wF']
			bV = param_dict[key]['bV']
			wV = param_dict[key]['wV']
		else:
			raise ValueError("Dictionary with len(keys)==1 is not 'rbm'.")
	elif len(keys) == 2:
		if keys[0] == 'rbm_cnn/~/F' and keys[1] == 'rbm_cnn/~/V':
			bF = param_dict[keys[0]]['b']
			wF = param_dict[keys[0]]['w']
			wF = einops.rearrange(wF, ' a b c d -> (a b c d)')
			bV = param_dict[keys[1]]['b']
			wV = param_dict[keys[1]]['w']
			wV = einops.rearrange(wV, ' a b c d -> (a b c d)')
		elif keys[0] == 'rbm_cnn_2/~/F' and keys[1] == 'rbm_cnn_2/~/V':
			bF = param_dict[keys[0]]['b']
			wF = param_dict[keys[0]]['w']
			wF = einops.rearrange(wF, ' a b c d -> (a b c d)')
			bV = param_dict[keys[1]]['b']
			wV = param_dict[keys[1]]['w']
			wV = einops.rearrange(wV, ' a b c d -> (a b c d)')			
	else:
		raise ValueError("Dictionary has more than two keys")
	if np_only: 
		F_array = np.concatenate((bF, wF))
		V_array = np.concatenate((bV, wV))
	else:
		F_array = jnp.concatenate((bF, wF))
		V_array = jnp.concatenate((bV, wV))
	return F_array, V_array	
	# return jax.device_get(F_array), jax.device_get(V_array)

def similarity_fn(w1, w2):
	"""
	Takes param dictionaries w1, w2, return similarity
	"""
	# print(type(w1))
	def _S_max(array1, array2):
		w_difference_cos = jnp.array([jnp.cos(2. * (array1 - array2)), jnp.cos(2. * (-array1 - array2))])
		# print(w_difference_cos.shape)
		w_cos_mean = jnp.mean(w_difference_cos, axis=1)
		w_cos_max = jnp.max(w_cos_mean, axis=0)
		# print(w_cos_max.shape)
		return jnp.mean(w_cos_max)
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = _S_max(F_array_1, F_array_2)
	V_max = _S_max(V_array_1, V_array_2)
	return (1. + (F_max + V_max) / 2.) /2. 

def similarity_fn_np(w1, w2):
	"""
	Takes param dictionaries w1, w2, return similarity
	"""
	# print(type(w1))
	def _S_max(array1, array2):
		w_difference_cos = np.array([np.cos(2. * (array1 - array2)), np.cos(2. * (-array1 - array2))])
		# print(w_difference_cos.shape)
		w_cos_mean = np.mean(w_difference_cos, axis=1)
		w_cos_max = np.max(w_cos_mean, axis=0)
		# print(w_cos_max.shape)
		return np.mean(w_cos_max)
	F_array_1, V_array_1 = extract_V_F_params(w1, True)
	F_array_2, V_array_2 = extract_V_F_params(w2, True)
	F_max = _S_max(F_array_1, F_array_2)
	V_max = _S_max(V_array_1, V_array_2)
	return (1. + (F_max + V_max) / 2.) /2. 

def similarity_fn_exp(w1, w2, epsilon0):
	"""
	Takes param dictionaries w1, w2, return similarity
	"""
	# print(type(w1))
	def _S_max(array1, array2):
		w_difference_cos = jnp.array([jnp.cos(2. * (array1 - array2)), jnp.cos(2. * (-array1 - array2))])
		# print(w_difference_cos.shape)
		w_cos_mean = jnp.mean(w_difference_cos, axis=1)
		w_cos_max = jnp.max(w_cos_mean, axis=0)
		# print(w_cos_max.shape)
		return jnp.mean(w_cos_max)
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = _S_max(F_array_1, F_array_2)
	V_max = _S_max(V_array_1, V_array_2)
	similarity = (1. + (F_max + V_max) / 2.) /2. 	
	return jnp.exp(-(jnp.ones_like(similarity) - similarity) / (2. * epsilon0))

# def euclidean_norm(w1, w2):
# 	"""
# 	Takes param dictionaries w1, w2, return euclidean norm
# 	"""
# 	F_array_1, V_array_1 = extract_V_F_params(w1)
# 	F_array_2, V_array_2 = extract_V_F_params(w2)
# 	F_max = jnp.linalg.norm(F_array_1 - F_array_2) 
# 	V_max = jnp.linalg.norm(V_array_1 - V_array_2) 
# 	return F_max + V_max

def gram_matrix(data):
  dot_fn = lambda X, Y:  sum(jax.tree_leaves(jax.tree_map(lambda x, y: jnp.sum(jnp.multiply(x, y)), X, Y)))
  doc_fn_vec = jax.vmap(jax.vmap(dot_fn, (0, None)), (None, 0))
  return np.array(doc_fn_vec(data, data))

def euclidean_norm(w1, w2):
  """
  Takes param dictionaries w1, w2, return euclidean norm
  """
  F_array_1, V_array_1 = extract_V_F_params(w1)
  F_array_2, V_array_2 = extract_V_F_params(w2)
  array1 = jnp.concatenate([F_array_1, V_array_1])
  array2 = jnp.concatenate([F_array_2, V_array_2])
  return jnp.linalg.norm(array1 - array2) 

def euclidean_mod_pi(w1, w2):
	"""
	Takes param dictionaries w1, w2, return euclidean norm
	"""
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = jnp.linalg.norm(F_array_1 - F_array_2)
	V_max = jnp.linalg.norm(V_array_1 - V_array_2) 
	return jnp.mod(F_max, np.pi) + jnp.mod(V_max, np.pi)

def difference_img(w1, w2):
	"""
	Takes param dictionaries w1, w2, return euclidean norm
	"""
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = (F_array_1 - F_array_2) / (np.pi/2)
	V_max = (V_array_1 - V_array_2) / (np.pi/2)
	print(F_max.shape)
	return jnp.concatenate([F_max, V_max])[:, 0, 0]	

def similarity_fn_test(w1, w2):
	"""
	Takes param dictionaries w1, w2, return similarity
	"""
	# print(type(w1))
	def _S_max(array1, array2):
		w_difference_cos = jnp.cos(2. * (array1 - array2))
		# print(w_difference_cos.shape)
		# w_cos_mean = jnp.mean(w_difference_cos, axis=1)
		# w_cos_max = jnp.max(w_cos_mean, axis=0)
		# print(w_cos_max.shape)
		return jnp.mean(w_difference_cos)
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = _S_max(F_array_1, F_array_2)
	V_max = _S_max(V_array_1, V_array_2)
	return (1. + (F_max + V_max) / 2.) /2. 

def similarity_local_patch(w1, w2):
	def _S_avg(array1, array2):
		my_iter = itertools.product([-1, 1], repeat=4)
		basis_list = []
		for basis in my_iter:
		  basis_list.append([1.]+list(basis))
		basis_array = jnp.array(basis_list)
		# print(array1.shape)
		w_difference = jnp.tensordot(basis_array, (array1 - array2), axes=1)
		print(f"w_difference shape is {w_difference.shape}")
		w_difference_cos = jnp.cos(w_difference)
		w_difference_cos_mean = jnp.mean(w_difference_cos, axis=0)
		print(f"w_difference_cos_mean shape is {w_difference_cos_mean.shape}")
		return jnp.mean(w_difference_cos_mean)
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_max = _S_avg(F_array_1, F_array_2)
	V_max = _S_avg(V_array_1, V_array_2)
	return (1. + (F_max + V_max) / 2.) /2. 	

def similarity_quasilocal(w1, w2):
	def _S_avg(array1, array2):
		my_iter = itertools.product([-1, 1], repeat=4)
		basis_list = []
		for basis in my_iter:
		  basis_list.append([1.]+list(basis))
		basis_array = jnp.array(basis_list)
		# print(array1.shape)
		w_difference = jnp.tensordot(basis_array, (array1 - array2), axes=1)
		print(f"w_difference shape is {w_difference.shape}")
		w_difference_cos = jnp.cos( w_difference)
		w_difference_cos_mean = jnp.mean(w_difference_cos, axis=0)
		print(f"w_difference_cos_mean shape is {w_difference_cos_mean.shape}")
		return w_difference_cos_mean
	F_array_1, V_array_1 = extract_V_F_params(w1)
	F_array_2, V_array_2 = extract_V_F_params(w2)
	F_patch1 = _S_avg(F_array_1, F_array_2)
	F_patch2 = jnp.roll(F_patch1, -1, axis=1)
	V_patch1 = _S_avg(V_array_1, V_array_2)
	V_patch2 = jnp.roll(V_patch1, 1, axis=0)
	patch_prod = F_patch1 * F_patch2 * V_patch1 * V_patch2
	print(f'patch prod shape is {patch_prod.shape}')
	print(f"modifed cos")
	return jnp.mean(patch_prod)

def similarity_modified(w1, w2):
  def _S_max(array1, array2):
    # return jnp.cos(jnp.mod((array1 - array2), np.pi/2.) )
    # return jnp.cos((array1 - array2)/( np.pi/2.) )
    # return jnp.abs((jnp.mod((array1 - array2), np.pi) - np.pi/2)*( 4/np.pi)) - 1.
    f = lambda x: jnp.cos((jnp.mod( x + np.pi /2, np.pi ))  ) + 1 * jnp.sign(jnp.cos(x ))
    value1 = jnp.where(jnp.mod((array1 - array2), np.pi)<np.pi/2, f(2 * (array1 - array2)), f(-2 *(array1 - array2)) )
    value2 = jnp.where(jnp.mod((-array1 - array2), np.pi)<np.pi/2, f(2 * (-array1 - array2)), f(-2 *(-array1 - array2)) )
    return jnp.amax(jnp.stack([jnp.mean(value1, axis=0), jnp.mean(value2, axis=0)]), axis=0)
  F_array_1, V_array_1 = extract_V_F_params(w1)
  F_array_2, V_array_2 = extract_V_F_params(w2)
  F_patch1 = _S_max(F_array_1, F_array_2)
  V_patch1 = _S_max(V_array_1, V_array_2)
  return (jnp.mean(F_patch1) + jnp.mean(V_patch1)) * 0.5

def similarity_modified_2(w1, w2):
  def _S_max(array1, array2):
    f = lambda x: jnp.cos((jnp.mod( x + np.pi /2, np.pi ))  ) + 1 * jnp.sign(jnp.cos(x ))
    value1 = jnp.where(jnp.mod((array1 - array2), np.pi)<np.pi/2, 
                       f(2 * (array1 - array2))**2 * jnp.sign(f(2 * (array1 - array2))), 
                       f(-2 *(array1 - array2))**2 * jnp.sign(f(-2 *(array1 - array2))) )
    value2 = jnp.where(jnp.mod((-array1 - array2), np.pi)<np.pi/2,
                       f(2 * (-array1 - array2))**2 * jnp.sign(f(2 * (-array1 - array2))), 
                       f(-2 *(-array1 - array2))**2 * jnp.sign(f(-2 *(-array1 - array2))) )
    return jnp.amax(jnp.stack([jnp.mean(value1, axis=0), jnp.mean(value2, axis=0)]), axis=0)
  F_array_1, V_array_1 = extract_V_F_params(w1)
  F_array_2, V_array_2 = extract_V_F_params(w2)
  F_patch1 = _S_max(F_array_1, F_array_2)
  V_patch1 = _S_max(V_array_1, V_array_2)
  return (jnp.mean(F_patch1) + jnp.mean(V_patch1)) * 0.5

def similarity_nn(w1, w2, model_name='rbm_noise'):
  """Compare two weights w1 and w2 using a nearest neighbour similarity function. 
  The last two dimensions of w1 and w2 are assumed to be spin_shape[0] /2 and 
  spin_shape[1]. 
  Args:
    w1: pytree for weights w1.
    w2: pytree for weights w2.
    model_name: default is `rbm_noise` model.

  Returns:
    Float quantifies similarity.
  """  
  # assert model_name in w1.keys(), f"Model is not {model_name}."
  shape = jax.tree_leaves(utils.shape_structure(w1))[-2:]
  num_spins = 2 * shape[0] * shape[1]
  spin_cube_list = []
  for i in range(num_spins):
    spin_1d = np.zeros((num_spins, ))
    spin_1d[i] = 1.
    spin_2d = einops.rearrange(spin_1d, '(x y) -> x y', y=shape[1])
    spin_cube_F, spin_cube_V = tc_utils.stack_F_V_img(spin_2d)
    spin_cube_list.append(spin_cube_F)

  spin_cube_array = jnp.array(spin_cube_list)
  w1F = w1[model_name]['wF']
  w2F = w2[model_name]['wF']
  b1F = w1[model_name]['bF']
  b2F = w2[model_name]['bF']  
  # w1_summed = jax.tree_map(lambda x: jnp.sum(w1F * x), spin_cube_list)
  # w2_summed = jax.tree_map(lambda x: jnp.sum(w2F * x), spin_cube_list)
  # w_diff = jax.tree_map(lambda x, y: jnp.cos(x - y), w1_summed, w2_summed)
  # return jnp.mean(jnp.array(w_diff))
  w1_summed = jnp.sum(w1F * spin_cube_array, (1, 2, 3))
  w2_summed = jnp.sum(w2F * spin_cube_array, (1, 2, 3))
  wF_term = jnp.sum(jnp.cos(2 * (w1_summed - w2_summed))) / num_spins
  # bF_term = jnp.mean(jnp.cos(2 * (b1F - b2F)))
  bF_term = jnp.cos(jnp.mean(2 * (b1F - b2F)))
  return (wF_term + bF_term) / 2.

def similarity_nn_prod(w1, w2, model_name='rbm_noise'):
  """Compare two weights w1 and w2 using a nearest neighbour similarity function. 
  The last two dimensions of w1 and w2 are assumed to be spin_shape[0] /2 and 
  spin_shape[1]. 
  Args:
    w1: pytree for weights w1.
    w2: pytree for weights w2.
    model_name: default is `rbm_noise` model.

  Returns:
    Float quantifies similarity.
  """  
  # assert model_name in w1.keys(), f"Model is not {model_name}."
  shape = jax.tree_leaves(utils.shape_structure(w1))[-2:]
  num_spins = 2 * shape[0] * shape[1]
  spin_cube_list = []
  for i in range(num_spins):
    spin_1d = np.zeros((num_spins, ))
    spin_1d[i] = 1.
    spin_2d = einops.rearrange(spin_1d, '(x y) -> x y', y=shape[1])
    spin_cube_F, spin_cube_V = tc_utils.stack_F_V_img(spin_2d)
    spin_cube_list.append(spin_cube_F)

  spin_cube_array = jnp.array(spin_cube_list)
  w1F = w1[model_name]['wF']
  w2F = w2[model_name]['wF']
  b1F = w1[model_name]['bF']
  b2F = w2[model_name]['bF']  
  # w1_summed = jax.tree_map(lambda x: jnp.sum(w1F * x), spin_cube_list)
  # w2_summed = jax.tree_map(lambda x: jnp.sum(w2F * x), spin_cube_list)
  # w_diff = jax.tree_map(lambda x, y: jnp.cos(x - y), w1_summed, w2_summed)
  # return jnp.mean(jnp.array(w_diff))
  w1_weighted = w1F * spin_cube_array
  w2_weighted = w2F * spin_cube_array
  w1_replaced = jnp.where(w1_weighted == 0., 1., w1_weighted)
  w2_replaced = jnp.where(w2_weighted == 0., 1., w2_weighted)
  w1_prod = jnp.product(w1_replaced, axis=(1, 2, 3))
  w2_prod = jnp.product(w2_replaced, axis=(1, 2, 3))
  wF_term = jnp.sum(jnp.cos(4 * (w1_prod - w2_prod))) / num_spins
  # bF_term = jnp.mean(jnp.cos(2 * (b1F - b2F)))
  bF_term = jnp.cos(jnp.mean(2 * (b1F - b2F)))
  # return (wF_term + bF_term) / 2.  
  return wF_term

# def similarity_fn_prod(w1, w2):
# 	"""
# 	Takes param dictionaries w1, w2, return similarity
# 	"""
# 	def _S_max(array1, array2):
# 		w_difference_cos = jnp.array([jnp.cos(2. * (array1 - array2)), jnp.cos(2. * (-array1 - array2))])
# 		w_cos_mean = jnp.mean(w_difference_cos, axis=1)
# 		w_cos_max = jnp.max(w_cos_mean, axis=0)
# 		return jnp.prod(w_cos_max)
# 	F_array_1, V_array_1 = extract_V_F_params(w1)
# 	F_array_2, V_array_2 = extract_V_F_params(w2)
# 	F_max = _S_max(F_array_1, F_array_2)
# 	V_max = _S_max(V_array_1, V_array_2)
# 	return (1. + (F_max + V_max) / 2.) /2. 	

def kernel_fn(similarity, epsilon):
	# return jnp.exp(-(1. - similarity) / (2. * epsilon))
	return jnp.exp(-(jnp.ones_like(similarity) - similarity) / (2. * epsilon))

def kernel_fn_np(similarity, epsilon):
	# return jnp.exp(-(1. - similarity) / (2. * epsilon))
	return np.exp(-(np.ones_like(similarity) - similarity) / (2. * epsilon))	

def kernel_fn_double_exp(similarity, epsilon1, epsilon2):
	# return jnp.exp(-(1. - similarity) / (2. * epsilon))
	kernel_exp =  jnp.exp(-(jnp.ones_like(similarity) - similarity) / (2. * epsilon1))	
	return jnp.exp(-(jnp.ones_like(kernel_exp) - kernel_exp) / (2. * epsilon2))	

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

def transition_mat(K_mat, return_z=False):
	"""Return both kernal matrix and a similar matrix A (which is symmetric)"""
	# K_mat = kernel_mat(all_w, epsilon)
	z1 = jnp.sum(K_mat, axis=1)
	z2 = jnp.sum(K_mat, axis=0)
	A_mat = K_mat / jnp.sqrt(jnp.outer(z1, z2))
	if return_z:
		# return A_mat, jnp.diag(1. / jnp.sqrt(z2))
		return A_mat, z2
	return A_mat

def transition_mat_np(K_mat, return_z=False):
	"""Return both kernal matrix and a similar matrix A (which is symmetric)"""
	# K_mat = kernel_mat(all_w, epsilon)
	z1 = np.sum(K_mat, axis=1)
	z2 = np.sum(K_mat, axis=0)
	A_mat = K_mat / np.sqrt(np.outer(z1, z2))
	if return_z:
		# return A_mat, jnp.diag(1. / jnp.sqrt(z2))
		return A_mat, z2
	return A_mat