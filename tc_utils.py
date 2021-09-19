#@title Toric Code utilities
import numpy as np
import jax
import operators
import bonds

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
          np.pi / 4. * np.array([-1., -1., 1., 1.])]
  bias_list = np.array([0., np.pi / 2., np.pi / 2., 0.])
  wF = weight_list[sector-1]
  bF = np.array([bias_list[sector-1]])

  return {'rbm':dict(bF=bF, wF=wF, bV=np.array([0.]), wV=np.zeros(4))}

def generate_uniform_noise_param(key, param_dict, amp):
  param_arrays, tree_def = jax.tree_flatten(param_dict)
  noise_values = [
      jax.random.uniform(k, p.shape, minval=-amp, maxval=amp)
      for k, p in zip(jax.random.split(key, len(param_arrays)), param_arrays)
  ]
  noise_values = jax.tree_unflatten(tree_def, noise_values)
  return jax.tree_map(lambda x, y: x + y, param_dict, noise_values)

def get_face_bonds(spin_shape):
	return bonds.create_bond_list(size=(spin_shape[0], spin_shape[1]), input_bond_list=[(0,0), (1, 0), (2, 0), (1,1)])

def get_vertex_bonds(spin_shape):
	return bonds.create_bond_list(size=(spin_shape[0], spin_shape[1]), input_bond_list=[(1,0), (2, 0), (3, 0), (2, spin_shape[0]-1)]) #fixed bug for odd lattices

def set_up_ham_field(spin_shape, h_z):
  #Set up hamiltonian in field h_z
  num_col =  spin_shape[1]
  num_row = spin_shape[0]
  num_spins = num_col * num_row
  # face_operator_bonds = bonds.create_bond_list(size=(num_row, num_col), input_bond_list=[(0,0), (1, 0), (2, 0), (1,1)])
  # vertex_operator_bonds = bonds.create_bond_list(size=(num_row, num_col), input_bond_list=[(1,0), (2, 0), (3, 0), (2, num_col-1)]) #fixed bug for odd lattices
  face_operator_bonds = get_face_bonds(spin_shape)
  vertex_operator_bonds = get_vertex_bonds(spin_shape) 
  pauli_operator_bonds = np.arange(0, num_spins, 1)
  # Define hamiltonian 
  myham = operators.ToricCodeHamiltonian(Jv=1., Jf=1., h = h_z, face_bonds = face_operator_bonds, vertex_bonds=vertex_operator_bonds, pauli_bonds=pauli_operator_bonds)  

  return myham