#@title Define Fixed Operator property
import jax
import jax.numpy as jnp


class FixedOperators():
  """
  Operator class with apply() method for getting relevant matrix elements for local energy. 
  """
  def apply(self, config, psi, model_params):
    "Return amplitude: <c|H|psi>"
    vectorized_psi = jax.vmap(psi, in_axes=(None, 0))
    new_configs, mat_eles = self.get_terms(config)
    psi_newconfigs = vectorized_psi(model_params, jnp.stack(new_configs, axis=0))
    mat_eles = jnp.stack(mat_eles, axis=0)
    return jnp.sum(psi_newconfigs * mat_eles )


#@title Define Class for Bond Operators

class FaceBond(FixedOperators):
  def __init__(self, config, Jf, bond):
    self.Jf = Jf
    self.config = config
    self.bond = bond
  
  def get_terms(self, config):
    "Return a list of new configurations and the matrix elements between them"
    config = self.config
    spin_prod = jnp.prod(config[self.bond])
    mat_ele = - self.Jf * spin_prod #mat ele is product of spins
    return (face_bond_sample(config, self.bond),) , (mat_ele,)

class VertexBond(FixedOperators):

  def __init__(self, config, Jv, bond):
    self.Jv = Jv
    self.config = config
    self.bond = bond
  
  def get_terms(self, config):
    "Return a list of new configurations and the matrix elements between them"
    config = self.config
    mat_ele = - self.Jv
    return (vertex_bond_sample(config, self.bond),) , (mat_ele,)   

class PauliBond(FixedOperators):
  '''
  Matrix elements and connected configurations for magnetic field in the Toric Code model. 
  '''
  def __init__(self, config, h, bond):
    self.hx, self.hy, self.hz = h
    self.config = config
    self.bond = bond
  
  def get_terms(self, config):
    "Return a list of new configurations and the matrix elements between them"
    config = self.config
    
    config_x = vertex_bond_sample(config, self.bond)
    config_y = vertex_bond_sample(config, self.bond)
    config_z = face_bond_sample(config, self.bond)

    mat_ele_x = self.hx
    mat_ele_y = self.hy * config[self.bond] * 0 #### set this to 0 for now
    mat_ele_z = self.hz * config[self.bond]
    return (config_z, ) , (mat_ele_z, )

class VertexBondExpModel(FixedOperators):
  """
  Bond for exponential term. 
  """
  def __init__(self, config, Jv, beta, bond):
    self.Jv = Jv
    self.betax, self.betay, self.betaz = beta
    self.config = config
    self.bond = bond
  
  def get_terms(self, config):
    "Return a list of new configurations and the matrix elements between them"
    config = self.config
    mat_ele = - self.Jv
    # spin_prod = self.betaz /2. * config[self.bond]
    # print(jnp.prod(self.self.betaz /2. *config[self.bond]))
    spin_prod = jnp.prod(jnp.exp(self.betaz /2. * config[self.bond]))
    # print(self.beta)
    mat_ele_exp = self.Jv * spin_prod #mat ele is product of spins
    # print(mat_ele_exp)
    # return (vertex_bond_sample(config, self.bond),config) , (mat_ele,mat_ele)
    return (vertex_bond_sample(config, self.bond), config, ) , (mat_ele, mat_ele_exp, ) 

#@title Define Class for Toric Code Hamiltonian
class ToricCodeHamiltonian(FixedOperators):
  def __init__(self, Jv, Jf, h, 
               face_bonds, vertex_bonds, pauli_bonds):
    self.Jv = Jv
    self.Jf = Jf
    self.h = h
    self.face_bonds = face_bonds
    self.vertex_bonds = vertex_bonds
    self.pauli_bonds = pauli_bonds
    self.face_bond_op_list = []
    self.vertex_bond_op_list = []
    self.pauli_bond_op_list = []
    for face_bond in self.face_bonds:     #Loop through all face bonds, get new configurations and mat elements
      face_bond_op = FaceBond(self.Jf, face_bond)
      self.face_bond_op_list.append(face_bond_op)
    for vertex_bond in self.vertex_bonds:
      vertex_bond_op = VertexBond(self.Jv, vertex_bond)
      self.vertex_bond_op_list.append(vertex_bond_op)
    for pauli_bond in self.pauli_bonds:
      pauli_bond_op = PauliBond(self.h, pauli_bond)  
      self.pauli_bond_op_list.append(pauli_bond_op)    
  def get_terms(self, config, operator_params=None):
    vertex_list = []    #List for all terms from vertex operators
    face_list = []      #List for all terms from face operators
    pauli_list = []     #List for all terms from pauli operators
    for face_bond_op in self.face_bond_op_list:
      face_list.append(face_bond_op.get_terms(config))
    for vertex_bond_op in self.vertex_bond_op_list:
      vertex_list.append(vertex_bond_op.get_terms(config))
    for pauli_bond_op in self.pauli_bond_op_list:
      pauli_list.append(pauli_bond_op.get_terms(config))
    all_terms_list = vertex_list + face_list + pauli_list
    cs, matrix_elements = zip(*all_terms_list)
    # print(jax.tree_leaves(matrix_elements))
    # print(len(jax.tree_leaves(cs)))
    # print(matrix_elements)
    return jax.tree_leaves(cs), jax.tree_leaves(matrix_elements)
    