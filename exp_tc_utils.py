#@title Exponential Toric Code model utilities
import numpy as np
import operators
import bonds

def set_up_expham_field(num_spins, h_z):
  #Set up hamiltonian
  num_col =  np.sqrt(num_spins // 2).astype(int)
  num_row = 2 * num_col
  face_operator_bonds = bonds.create_bond_list(size=(num_row, num_col), input_bond_list=[(0,0), (1, 0), (2, 0), (1,1)])
  vertex_operator_bonds = bonds.create_bond_list(size=(num_row, num_col), input_bond_list=[(1,0), (2, 0), (3, 0), (2, num_col-1)]) #fixed bug for odd lattices
  # Define hamiltonian 
  myham = operators.ExpHamiltonian(Jv=1., Jf=1., h = h_z, face_bonds = face_operator_bonds, vertex_bonds=vertex_operator_bonds)  

  return myham