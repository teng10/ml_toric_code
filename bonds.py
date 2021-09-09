#@title Create Bond Lists

import numpy as np

def create_bond_list(size, lattice_vectors = {(0, 1), (1,0)},  
                     input_bond_list=[(0,0), (1, 0), (2, 0), (1,1)], translations=[(2, 0), (0, 1)]):
  "Return a list of bonds given a specified single bond and translations"
  c_vector = np.arange(0, size[0] * size[1], 1)
  c_graph = np.reshape(c_vector, size)
  # print(f"The lattice is {c_graph}") 

  all_bond_list = [] # a list for collecting all the bonds
  all_bond_list.append(input_bond_list)
  i = 0 # create an index tracking all possible translations
  translate_list = np.copy(all_bond_list)
  for translate in translations: #loop through operations in translations
    for single_bond_list in translate_list: # for each object in first tranlation, do the second translation
      while single_bond_list[0][i] / (size[i]  - translate[i]) < 1: #enforcing PBC
        new_bond_list = []
        for idx in single_bond_list:
          ele = tuple(map(lambda x, y: x + y, idx, translate))
          new_bond_list.append(ele)
        # print(f"new bond list is {new_bond_list}")
        single_bond_list = new_bond_list
        # print(f"singe bond list is changed to  {single_bond_list}")
        # print(f"single bond list mod is {single_bond_list[0][i] % size[i]}")
        all_bond_list.append(single_bond_list)
    translate_list =  np.copy(all_bond_list) #save a copy of the elements after first tranlation
    i+=1 #loop through next translations
    
  final_indices = []
  for bond_list in all_bond_list:
    two_d_indices = list(zip(*bond_list)) # list of two 4-tuples
    two_d_indices_a = tuple(x % (size[0] ) for x in two_d_indices[0]) # mod the first tuple with size[0]
    two_d_indices_b = tuple(x % (size[1] ) for x in two_d_indices[1]) # mod the first tuple with size[1]
    two_d_indices = (two_d_indices_a, two_d_indices_b) # get the new 2d tuples
    # two_d_indices = [two_d_indices_a, two_d_indices_b] # get the new 2d tuples
    # print(two_d_indices)
    final_indices.append(c_graph[two_d_indices]) #indices for operators
  # print(f"The list of bonds for {input_bond_list} is")
  # print(f"{final_indices}")
  return np.asarray(final_indices)