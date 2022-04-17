#@title Toric Code utilities
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import operators
import bonds
import sample_utils
import einops
import utils
import itertools

def get_bias(sector):
  bias_list = [0., np.pi / 2., np.pi / 2., 0.]
  return bias_list[sector-1]

def get_weights(sector):
  weight_list = [np.pi / 4. * np.array([[1., 0.], [1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[1., 0.], [-1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[-1., 0.], [1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[-1., 0.], [-1., 1.], [1., 0.]])]
  return weight_list[sector-1]

def get_rbm_params(sector):
  weight_list = [np.pi / 4. * np.array([1., 1., 1., 1.]), 
          np.pi / 4. * np.array([1., -1., 1., 1.]), 
          np.pi / 4. * np.array([-1., 1., 1., 1.]), 
          np.pi / 4. * np.array([-1., -1., 1., 1.]), 
          np.array([0., 0., 0., 0.])]
  bias_list = np.array([0., np.pi / 2., np.pi / 2., 0., 0.])
  wF = weight_list[sector-1]
  bF = np.array([bias_list[sector-1]])

  return {'rbm':dict(bF=bF, wF=wF, bV=np.array([0.]), wV=np.zeros(4))}

def get_cnn_params(sector):
  weight_list = [np.pi / 4. * np.array([[1., 0.], [1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[1., 0.], [-1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[-1., 0.], [1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[-1., 0.], [-1., 1.], [1., 0.]])]
  bias_list = np.array([0., np.pi / 2., np.pi / 2., 0., 0.])
  wF = np.expand_dims(np.expand_dims(weight_list[sector-1], -1), -1)
  bF = np.array([bias_list[sector-1]])

  return {'rbm_cnn/~/F':dict(b=bF, w=wF), 'rbm_cnn/~/V':dict(b=np.array([0.]), w=np.zeros((3, 2, 1, 1)))}    

def get_cnn_channel_params(sector, channel):
  weight_list = [np.pi / 4. * np.array([[1., 0.], [1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[1., 0.], [-1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[-1., 0.], [1., 1.], [1., 0.]]), 
              np.pi / 4. *  np.array([[-1., 0.], [-1., 1.], [1., 0.]])]
  bias_list = np.array([0., np.pi / 2., np.pi / 2., 0., 0.])
  wF = np.expand_dims(np.expand_dims(weight_list[sector-1], -1), -1)
  wzero = np.zeros_like(wF)
  bF = np.array([bias_list[sector-1]])
  bzero = np.zeros_like(bF)
  bF = np.concatenate([bF, bzero], -1)
  wF = np.concatenate([wF, wzero], -1)
  bV = np.concatenate([bzero, bzero], -1)
  wV = np.concatenate([wzero, wzero], -1)

  return {'rbm_cnn/~/F':dict(b=bF, w=wF), 'rbm_cnn/~/V':dict(b=bV, w=wV)}      

def get_params_zeeman():
  b = np.array([np.pi / 4.])
  w = np.array([np.pi / 4., 0., 0., 0.])

  return {'rbm':dict(bF=b, wF=w, bV=b, wV=w)}  

def convert_rbm_expanded(dict_rbm, shape, old_name = 'rbm', new_name='rbm_noise'):
  """
  Utility function converting RBM parameter dictionary to expanded paramter dictionary
  """
  def _tile_param(param_array):
    param_expanded = jnp.expand_dims(param_array, (-1, -2))
    return jnp.tile(param_expanded, (1, shape[0], shape[1]))
  new_dict = jax.tree_map(_tile_param, dict_rbm)
  return {new_name: new_dict[old_name]}

# def generate_single_noise_param(key, param_dict, amp, return_noise=False):
#   key_local, key_local_2 = jax.random.split(key, 2)
#   param_arrays, tree_def = jax.tree_flatten(param_dict)
#   # noise_values = jax.tree_unflatten(tree_def, noise_values)
#   noise_local = [jnp.zeros_like(param) for param in param_arrays]
#   index = jax.random.randint(key_local, shape=(), minval=0, maxval=len(noise_local))  
#   noise_shape = noise_local[index].shape
#   key1, key2, key3 = jax.random.split(key_local_2, 3)
#   index1 = jax.random.randint(key1, shape=(), minval=0, maxval=noise_shape[0])
#   index2 = jax.random.randint(key2, shape=(), minval=0, maxval=noise_shape[1])
#   index3 = jax.random.randint(key3, shape=(), minval=0, maxval=noise_shape[2])
#   new_noise_loc = noise_local[index].at[index1, index2, index3].set(amp)
#   noise_local[index] = new_noise_loc
#   # noise = [amp * local_index for local_index in noise_local]
#   noise = jax.tree_unflatten(tree_def, noise_local) 
#   if return_noise:
#   	return jax.tree_map(lambda x, y: x + y, param_dict, noise), noise
#   return jax.tree_map(lambda x, y: x + y, param_dict, noise)

def generate_single_noise_param(key, param_dict, amp_noise, return_noise=False):
  pack_fn, unpack_fn = utils.get_pack_unpack_fns(param_dict)
  param_flattened = pack_fn(param_dict)
  index = jax.random.randint(key, shape=(), minval=0, maxval=param_flattened.shape[0]) 
  zeros_flattened = jnp.zeros_like(param_flattened)
  noise_flattened = zeros_flattened.at[index].set(amp_noise)
  param_noise_flattened = param_flattened + noise_flattened
  # print(param_flattened.shape)
  if return_noise:
    return unpack_fn(param_noise_flattened), unpack_fn(noise_flattened)
  return unpack_fn(param_noise_flattened)

def generate_uniform_noise_param(key, param_dict, amp, local_updates=False, return_noise=False):
  key_noise, key_local = jax.random.split(key, 2)
  param_arrays, tree_def = jax.tree_flatten(param_dict)
  noise_values = [
      jax.random.uniform(k, p.shape, minval=-amp, maxval=amp)
      for k, p in zip(jax.random.split(key_noise, len(param_arrays)), param_arrays)
  ]
  noise_values = jax.tree_unflatten(tree_def, noise_values)
  if return_noise:
    return jax.tree_map(lambda x, y: x + y, param_dict, noise_values), noise_values
  return jax.tree_map(lambda x, y: x + y, param_dict, noise_values)

def generate_local_noise_param(key, param_dict, amp_noise, return_flips=False):
  model_name = 'rbm_noise'
  assert 'rbm_noise' in param_dict.keys(), "Model is not 'rbm_noise'."
  shape = jax.tree_leaves(utils.shape_structure(param_dict))[-2:]
  num_spins = 2 * shape[0] * shape[1]
  config = jnp.zeros(shape=(num_spins,))
  key_spin, key_noise = jax.random.split(key, 2)
  spin_flip = jax.random.randint(key_spin, shape=(1,), minval=0, maxval=num_spins)
  noise = jax.random.uniform(key_noise, (), minval=-amp_noise, maxval=amp_noise)
  # config_flipped = jax.ops.index_update(config, spin_flip, noise)
  config_flipped = jnp.asarray(config).at[spin_flip].set(noise)
  config_2d = einops.rearrange(config_flipped, '(x y) -> x y', x=shape[0]*2, y=shape[1])
  x_facebond, x_vertexbond = stack_F_V_img(config_2d)
  assert x_facebond.shape ==param_dict[model_name]['wF'].shape, "Spin flips and wF have different shape."
  # wF_updated = param_dict[model_name]['wF'] * x_facebond
  # wV_updated = param_dict[model_name]['wV'] * x_vertexbond
  # param_dict[model_name]['wF'] = wF_updated
  # param_dict[model_name]['wV'] = wV_updated
  ones_b = jnp.zeros((1, shape[0], shape[1]))
  flip_dict = {model_name:dict(bF=ones_b, wF=x_facebond, bV=ones_b, wV=x_vertexbond)}
  if return_flips:
    return jax.tree_map(lambda x, y: x + y, param_dict, flip_dict), spin_flip
  return jax.tree_map(lambda x, y: x + y, param_dict, flip_dict)

def generate_FV_noise_param(key, param_dict, amp_noise, return_noise=False):
  model_name = 'rbm_noise'
  assert 'rbm_noise' in param_dict.keys(), "Model is not 'rbm_noise'."
  shape = jax.tree_leaves(utils.shape_structure(param_dict))[-2:]
  key_noise, key_local = jax.random.split(key, 2)
  param_arrays, tree_def = jax.tree_flatten(param_dict)
  rngs = jax.random.split(key_noise, len(param_arrays))
  noise_values = [
      jax.random.uniform(k, [p.shape[0]], minval=-amp_noise, maxval=amp_noise)
      for k, p in zip(rngs, param_arrays)
  ]
  random_location = jax.random.randint(key_local, (2,), minval=0, maxval=shape[0]-1)
  modified_params = [param.at[..., random_location[0], random_location[1]].add(noise) for param, noise in zip(param_arrays, noise_values)]
  noise_dict = jax.tree_unflatten(tree_def, noise_values)
  param_dict = jax.tree_unflatten(tree_def, modified_params)
  if return_noise:
    return param_dict, noise_dict
  return param_dict

def generate_m_particles_param(key, param_dict, return_flips=False):
  model_name = 'rbm_noise'
  assert 'rbm_noise' in param_dict.keys(), "Model is not 'rbm_noise'."
  shape = jax.tree_leaves(utils.shape_structure(param_dict))[-2:]
  num_spins = 2 * shape[0] * shape[1]
  config = jnp.ones(shape=(num_spins,))
  spin_flip = jax.random.randint(key, shape=(1,), minval=0, maxval=num_spins)
  config_flipped = sample_utils.spin_flip_sampling(config, spin_flip)
  config_2d = einops.rearrange(config_flipped, '(x y) -> x y', x=shape[0]*2, y=shape[1])
  x_facebond, x_vertexbond = stack_F_V_img(config_2d)
  assert x_facebond.shape ==param_dict[model_name]['wF'].shape, "Spin flips and wF have different shape."
  # wF_updated = param_dict[model_name]['wF'] * x_facebond
  # wV_updated = param_dict[model_name]['wV'] * x_vertexbond
  # param_dict[model_name]['wF'] = wF_updated
  # param_dict[model_name]['wV'] = wV_updated
  ones_b = jnp.ones((1, shape[0], shape[1]))
  flip_dict = {model_name:dict(bF=ones_b, wF=x_facebond, bV=ones_b, wV=x_vertexbond)}
  if return_flips:
    return jax.tree_map(lambda x, y: x * y, param_dict, flip_dict), spin_flip
  return jax.tree_map(lambda x, y: x * y, param_dict, flip_dict)

def propose_param_fn(key, param_dict, p_mpar, amp_noise):
  """Propose with probabilty p_mpar m particle excitations; otherwise random noise. 

  Returns:
  array representing a packed `tree` and `unpack` function.
  """ 
  key_flip, key_propose = jax.random.split(key, 2)
  random_num = jax.random.uniform(key_flip)
  condition = random_num < p_mpar
  return jax.lax.cond(condition, lambda _: generate_m_particles_param(key_propose, param_dict), 
    lambda _: generate_FV_noise_param(key_propose, param_dict, amp_noise), None)  

def get_face_bonds(spin_shape):
	return bonds.create_bond_list(size=(spin_shape[0], spin_shape[1]), input_bond_list=[(0,0), (1, 0), (2, 0), (1,1)])

def get_vertex_bonds(spin_shape):
	return bonds.create_bond_list(size=(spin_shape[0], spin_shape[1]), input_bond_list=[(1,0), (2, 0), (3, 0), (2, spin_shape[0]-1)]) #fixed bug for odd lattices

def set_up_ham_field(spin_shape, h_z, Jv=1., Jf=1.):
  #Set up hamiltonian in field h_z
  num_col =  spin_shape[1]
  num_row = spin_shape[0]
  num_spins = num_col * num_row
  face_operator_bonds = get_face_bonds(spin_shape)
  vertex_operator_bonds = get_vertex_bonds(spin_shape) 
  pauli_operator_bonds = np.arange(0, num_spins, 1)
  # Define hamiltonian 
  myham = operators.ToricCodeHamiltonian(Jv=Jv, Jf=Jf, h = h_z, face_bonds = face_operator_bonds, vertex_bonds=vertex_operator_bonds, pauli_bonds=pauli_operator_bonds)  
  return myham

def set_up_ham_field_rotated(spin_shape, h, angle, Jv=1., Jf=1.):
  hz = h * jnp.cos(angle)
  hx = h * jnp.sin(angle)
  num_col =  spin_shape[1]
  num_row = spin_shape[0]
  num_spins = num_col * num_row
  face_operator_bonds = get_face_bonds(spin_shape)
  vertex_operator_bonds = get_vertex_bonds(spin_shape) 
  pauli_operator_bonds = jnp.arange(0, num_spins, 1)
  # Define hamiltonian 
  myham = operators.ToricCodeHamiltonianRotated(Jv=Jv, Jf=Jf, h = hz, hx = hx, face_bonds = face_operator_bonds, vertex_bonds=vertex_operator_bonds, pauli_bonds=pauli_operator_bonds)  
  return myham  

def stack_F_V_img(x_2d):
  """Roll 2d array x_2d to stacks of face and vertex.

  Args:
    array: array to be rolled.

  Returns:
    tuples of stacked arrays for face and vertex bonds.
  """
  x_r1 = jnp.roll(x_2d, -1, axis=0)
  # print(f"rolled 1 is {x_r1}")
  x_r2 = jnp.roll(x_2d, -2, axis=0)
  x_r3 = jnp.roll(x_r1, -1, axis=1)
  stacked_x = jnp.stack((x_2d, x_r1, x_r2, x_r3))
  # print(f"stacked x is {[stacked_x[:,::2, :][:,i,1] for i in range(3)]}") # 0-axis is stacked index
  #                                                                     # 1/2 axis are face operators 
  x_facebond = stacked_x[:,::2, :]
  # print(x_facebond[:, ...])
  x_r1 = jnp.roll(x_2d, -1, axis=0)
  # print(f"rolled 1 is {x_r1}")
  x_r2 = jnp.roll(x_r1, -1, axis=0)
  x_r3 = jnp.roll(x_r2, -1, axis=0)
  x_r4 = jnp.roll(x_r2, 1, axis=1)
  stacked_x = jnp.stack((x_r1, x_r2, x_r3, x_r4))
  # print(f"stacked x is {[stacked_x[:,::2, :][:,i,1] for i in range(3)]}") # 0-axis is stacked index
  #                                                                     # 1/2 axis are face operators 
  x_vertexbond = stacked_x[:,::2, :]
  return x_facebond, x_vertexbond

def set_partial_params_const(pytree, names_list, c, model_name='rbm_noise'):
  """Set paramsters in `names_list` to a constant `c` of the same shape. 
  The pytree is assumed to have the same structure as `rbm_noise` model.

  Args:
    pytree: pytree to be modified.
    names_list: a list of names of parameters to be modified.
    c: constant. 
    model_name: default is `rbm_noise` model.

  Returns:
    A new pytree with paramteres modified to constant `c`.
  """  
  # to-do: can we write a general function targeting parameters with certain name?
  # param_arrays, tree_def = jax.tree_flatten(pytree)
  # print(tree_def)
  # print(jax.tree_util.tree_structure(pytree))
  # print(jax.tree_util.treedef_tuple(tree_def))

  mutable_pytree = hk.data_structures.to_mutable_dict(pytree)
  for name in names_list:
    mutable_pytree[model_name][name] = jnp.zeros_like(pytree[model_name][name])

  return dict(mutable_pytree)
  # return hk.data_structures.to_haiku_dict(mutable_pytree)
 