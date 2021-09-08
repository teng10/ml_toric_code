#@title Plot utilities
import numpy as np
import seaborn
import matplotlib.pyplot as plt

def plot_weights(axarray, params, h_field, model_name):
	"""
	Plot weights for rbm ansatz
	"""
	bias_F =  params[model_name]['bF']
	bias_V =  params[model_name]['bV']
	weights_F = params[model_name]['wF'] / (np.pi / 4.)
	# print(weights_F.shape)
	weights_F = np.expand_dims(weights_F, 1)
	# weights_F = np.squeeze(np.squeeze(weights_F, -1), -1)
	weights_V = params[model_name]['wV'] / (np.pi / 4.)
	weights_V = np.expand_dims(weights_V, 1)
	# weights_V = np.squeeze(np.squeeze(weights_V, -1), -1)
	seaborn.heatmap(weights_F, ax=axarray[0], annot=True)
	axarray[0].set_title(f"Plaquette (pi/4), b={round_to_n(bias_F, 2)}")
	seaborn.heatmap(weights_V, ax=axarray[1], annot=True)
	axarray[1].set_title(f"Vertex, b={round_to_n(bias_V, 2)}") 


def plot_energies_field(num_spins, h_field, energy):  
  h_field_array = np.arange(h_field[0], h_field[1], h_field[2])
  plt.scatter(h_field_array, energy, label=f"N = {num_spins}")
  plt.plot(h_field_array, -h_field_array- 0.5, label='magnetic field')
  horiz_line_data = np.array([-1 for i in range(len(h_field_array))])
  plt.plot(h_field_array, horiz_line_data, '--', label='zero field energy')
  plt.legend()
  plt.xlabel("h_z")
  plt.ylabel("Energy <E> / N_spin")

def plot_energies_field_cluster(num_spins, h_field, energy, clusters=0):  
  h_field_array = np.arange(h_field[0], h_field[1], h_field[2])
  colormap = np.array(['r', 'b'])
  plt.scatter(h_field_array, energy, label=f"N = {num_spins}", c=colormap[clusters])
  plt.plot(h_field_array, -h_field_array - 0.5, label='magnetic field')
  horiz_line_data = np.array([-1 for i in range(len(h_field_array))])
  plt.plot(h_field_array, horiz_line_data, '--', label='zero field energy')
  plt.legend()
  plt.xlabel("h_z")
  plt.ylabel("Energy <E> / N_spin")