#@title notebook functions
import numpy as np
import jax
import jax.numpy as jnp
import utils
import itertools
import functools
import einops
import tc_utils
import mcmc_param
import estimates_mcmc
import datetime
import re
import pickle
from google.colab import files
import diffusion_map
from tqdm import tqdm
import scipy
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
import operators
import bonds

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def get_date():
  now = datetime.datetime.now()
  pattern = re.compile(r"-\d\d-\d\d")
  mo = pattern.search(str(now))
  return mo.group()[1:]
  
def save_file(data, file_name, file_path):
  now = datetime.datetime.now()
  pattern = re.compile(r"-\d\d-\d\d")
  mo = pattern.search(str(now))
  date = mo.group()[1:]
  file_name = f"{date}_" + file_name
  pickle.dump(data, open(file_path+file_name, 'wb'))
  files.download(file_path+file_name)

def get_color(xydata, ):
  """
  Map 2D wilson loop to color.
  """
  xydata = np.clip(xydata, -1, 1)
  key_xy_points = np.array([[0,0],[1,1],[1,-1],[-1,1], [-1,-1]],dtype=float)
  key_xy_RGBs = np.array([[1,1,1],[0,0,1], [1,0,0], [0, 1, 0], [0.8,0.4,0.]],dtype=float)
  reds = griddata(key_xy_points, key_xy_RGBs.T[0], xydata)
  greens = griddata(key_xy_points, key_xy_RGBs.T[1], xydata)
  blues = griddata(key_xy_points, key_xy_RGBs.T[2], xydata)
  return np.vstack((reds, greens, blues)).T

def kmeans_cluster(projections_dict, n_clusters):
  k_labels_dict = {}
  k_centers_dict = {}
  for key, projections in projections_dict.items():
    kmeans = KMeans(n_clusters).fit(projections)
    k_labels = kmeans.predict(projections)
    k_labels_dict[key] = k_labels
    k_centers_dict[key] = kmeans.cluster_centers_
  return k_labels_dict, k_centers_dict

def kmeans_cluster_dict(projections_dict, n_clusters_dict):
  k_labels_dict = {}
  k_centers_dict = {}
  for key, projections in projections_dict.items():
    n_clusters = n_clusters_dict[key]
    kmeans = KMeans(n_clusters).fit(projections)
    k_labels = kmeans.predict(projections)
    k_labels_dict[key] = k_labels
    k_centers_dict[key] = kmeans.cluster_centers_
  return k_labels_dict, k_centers_dict  

def construct_sec_labels(data_dict, sector_end):
  sector_label_dict = {}
  for key, data in data_dict.items():
      end_index = jax.tree_leaves(utils.shape_structure(data))[0]
      label_list = []
      for i in range(int(end_index/sector_end)):
        label_list.append(i * np.ones([sector_end]))
      sector_label_dict[key] = np.concatenate(label_list)
  return sector_label_dict

def get_kernel_fn_overlap(overlap_dict, key):
  """
  Return a kernel function (which returns precomputed overlap mat with corresponding key).
  """
  def _kernel_fn(data):
    return overlap_dict[key]
  return _kernel_fn

def get_kernel_normalized(data, kernel_fn, n_components=10):
  N_data = jax.tree_leaves(utils.shape_structure(data))[0]
  K = kernel_fn(data)
  Ns = np.ones((N_data)) / N_data
  K_normalized = K - Ns @ K - K @ Ns + Ns @ K @ Ns
  eigvals, eigens = scipy.linalg.eigh(K_normalized / N_data)
  eigvals_sorted = np.flip(eigvals, 0)[:n_components]
  eigens_sorted = np.flip(eigens, 1)[..., :n_components]
  projections = K_normalized @ eigens_sorted
  return eigvals_sorted, eigens_sorted, projections

def _screen_data_idx(eo_mat, sim_mat, epsilon1=0.1, epsilon2=0.2):
  ### Screen data based on comparison of overlap and similarity
  num_data = eo_mat.shape[0]
  indices_to_remove = set()
  for i in range(num_data):
    for j in range(i+1, num_data):
      if not(i in indices_to_remove or j in indices_to_remove):
        if abs(eo_mat[i, j] - 1.) <= epsilon1 and abs(sim_mat[i,j] - 1) >= epsilon2:
          indices_to_remove.add(j)
  all_indices = set(np.arange(num_data))          
  return list(all_indices - indices_to_remove)      

def _get_similarity_matrix(similarity_fn, params_stacked):
  S_vec = jax.vmap(similarity_fn, in_axes=(0, None))
  S_vec_vec = jax.vmap(S_vec, in_axes=(None, 0))
  S_mat = S_vec_vec(params_stacked, params_stacked)
  return  jax.device_get(S_mat)

def _get_overlap_matrix(vectors):
  num_vecs = len(vectors)
  indices_list = list(itertools.combinations_with_replacement(range(num_vecs), 2))
  overlap_mat = np.zeros((num_vecs, num_vecs))
  for index_pair in indices_list:
    overlap = exact_overlap(vectors[index_pair[0]], vectors[index_pair[1]])
    overlap_mat[index_pair] = overlap
    overlap_mat[index_pair[1], index_pair[0]] = overlap
  return overlap_mat

#@title overlap functions
def _update_s_mat(similarity_mat, 
               index_array, 
               value_array):
  dim = similarity_mat.shape
  raveled_index = jnp.ravel_multi_index(index_array.transpose(), dims=dim, mode='wrap')
  raveled_mat = similarity_mat.ravel()
  return raveled_mat.at[raveled_index].set(value_array).reshape(dim)

def include_overlaps(  
    indices, 
    similarity_mat,    
    stacked_params, 
    return_processed=False,
    estimate_overlap_fn=None, 
    ):
  """Returns a `similarity_matrix` with `percent` entries replaces with overlap estimation."""  
  sliced_params = utils.slice_along_axis(stacked_params, 0, indices)
  estimate_overlap_fn_vec = jax.vmap(estimate_overlap_fn)     #vmap estimate overlap function
  # estimate_overlap_jit = jax.jit(estimate_overlap_fn_vec)
  estimates_overlap_data = estimate_overlap_fn_vec(sliced_params)
  overlaps_est = estimates_overlap_data[0]
  overlaps_est = jnp.clip(overlaps_est, 0., 1.)
  similarity_mat_2 = _update_s_mat(similarity_mat, indices, overlaps_est)
  updated_similarity_mat = _update_s_mat(similarity_mat_2, jnp.flip(indices, axis=1), overlaps_est)
  if return_processed:
    return updated_similarity_mat, estimates_overlap_data
  return updated_similarity_mat

def get_vector(num_sites, batch_size, psi, psi_params):
  """Generates a full wavefunction by evaluating `psi` on basis elements."""
  
  def _get_full_basis_iterator(n_sites, batch_size):
    iterator = itertools.product([-1., 1.], repeat=n_sites)
    return _batch_iterator(iterator, batch_size)

  def _batch_iterator(iterator, batch_size=1):
    cs = []
    count = 0
    for c in iterator:
      cs.append(c)
      count += 1
      if count == batch_size:
        cs_out = np.stack(cs)
        cs = []
        count = 0
        yield cs_out
    if cs:
      yield np.stack(cs)

  psi_fn = jax.jit(jax.vmap(functools.partial(psi, psi_params)))
  psi_values = []
  basis_iterator = _get_full_basis_iterator(num_sites, batch_size)
  for cs in basis_iterator:
    psi_values.append(jax.device_get(psi_fn(cs)))
  return np.concatenate(psi_values)  


def include_overlaps_exact(  
    indices, 
    similarity_mat,    
    stacked_params, 
    return_processed=False,
    get_vector_fn=None, 

    ):
  """Returns a `similarity_matrix` with `percent` entries replaces with overlap estimation."""  
  similarity_mat = np.array(similarity_mat)
  sliced_params = utils.slice_along_axis(stacked_params, 0, indices)
  # print(utils.shape_structure(sliced_params))
  sliced_params = utils.split_axis(sliced_params, axis=0)
  overlap_list = []
  unique_indices = np.unique(indices)
  vec_list = list(np.zeros((similarity_mat.shape[0], )))
  for i in unique_indices:
    param = utils.slice_along_axis(stacked_params, 0, i)
    vector = get_vector_fn(param)
    vec_list[i] = vector
  for i, index_pair in enumerate(list(indices)):
    vector1 = vec_list[index_pair[0]]
    vector2 = vec_list[index_pair[1]]
    overlap = exact_overlap(vector1, vector2)
    similarity_mat[index_pair[0], index_pair[1]] = overlap
    similarity_mat[index_pair[1], index_pair[0]] = overlap
    overlap_list.append(overlap)
  if return_processed:
    return similarity_mat, (overlap_list, vec_list)
  return similarity_mat

def include_fake_ones_overlaps(
    indices,     
    similarity_mat,
    stacked_params,
    return_processed=False,
    get_vector_fn=None, 
):
  """Returns a `similarity_matrix` with `indices` entries replaces with 1.
  
  Args:
    indices: list of indices whoes entries to replace with 1. 
    similarity_matrix: original similartiy matrix.
    stacked_params: frozen param-dict with leading axis on all leaves
      corresponding to samples.
    return_processed: not used.
    get_vector_fn: not used.
  """
  similarity_mat = np.array(similarity_mat)
  sliced_params = utils.slice_along_axis(stacked_params, 0, indices)
  sliced_params = utils.split_axis(sliced_params, axis=0)
  for i, index_pair in enumerate(list(indices)):
    similarity_mat[index_pair[0], index_pair[1]] = 1.
    similarity_mat[index_pair[1], index_pair[0]] = 1.
  return similarity_mat

##### to-do change this!
def generate_field_noise_batch(key, 
                               params, 
                               noise_amp, 
                               batch, shape, 
                               return_noise=False):
  # rbm_exp_params = tc_utils.convert_rbm_expanded(params, (shape[0]//2, shape[1]))
  rngs = utils.split_key(key, (batch, 2))
  rbm_noise_params_batch = [tc_utils.generate_uniform_noise_param(rngs[i], params,  
                                                                  noise_amp, return_noise) for i in range(batch)]
  if return_noise:
    return list(map(list, zip(*rbm_noise_params_batch)))                                                                  
  return rbm_noise_params_batch

def generate_samples_T(key, h_field, spin_shape, 
                       len_chain_E, burn_E_factor, num_samples_E, 
                       psi_apply, 
                       len_chain, burn_factor, num_samples, 
                       T, noise_amp, 
                       p_flip,
                       params, 
                       angle, 
                       return_results=False, 
                       init_noise=0.):
  num_spins = spin_shape[0] * spin_shape[1]
  # Parameter for estimating energy
  len_chain_burn_E = burn_E_factor * num_spins
  # Parameters for mcmc_param
  len_chain_burn= burn_factor
  ham = tc_utils.set_up_ham_field_rotated(spin_shape, h_field, angle)
  estimate_ET_fn = functools.partial(estimates_mcmc.estimate_operator, operator=ham, psi_apply=psi_apply, 
                                    len_chain=len_chain_E, len_chain_first_burn=len_chain_burn_E, spin_shape=spin_shape, 
                                    num_samples=num_samples_E)
  key_sample, key_init, key_flip = jax.random.split(key, 3)
  rbm_noise_params_batch = generate_field_noise_batch(key_init, params,
                                                      init_noise, num_samples, spin_shape)
  # propose_move_param_fn = functools.partial(tc_utils.generate_FV_noise_param, amp_noise=noise_amp)
  propose_move_param_fn = functools.partial(tc_utils.propose_param_fn, p_mpar=p_flip, amp_noise=noise_amp)
  accept_rate, new_samples_dict, new_energies = mcmc_param.energy_sampling_mcmc(key_sample, rbm_noise_params_batch, 
                                                                  len_chain_burn, len_chain, estimate_ET_fn, 
                                                                  propose_move_param_fn, T, 
                                                                   return_results=return_results)

  new_samples = jax.tree_map(lambda x: einops.rearrange(x, ' a b c d e -> (a b) c d e'), new_samples_dict )
  new_energies = einops.rearrange(new_energies, ' a b -> (a b)') / num_spins
  return accept_rate, new_samples, new_energies    

def _diffusion_map_eigens(
    similarity_matrix,
    epsilon,
    vectors_and_kernel=False
):
  """Returns diffusion map eigenvalues with parameter `epsilon`.

  Args:
    similarity_matrix: matrix of size NxN...
    epsilon: parameter (noise?)
    vectors_and_kernel: whether to also return eigenvectors and kernel matrix.
  
  Returns:
    Eigenvalues of the diffusion map.
  """
  kernel_matrix = diffusion_map.kernel_fn(similarity_matrix, epsilon)
  if vectors_and_kernel:
    matrix_a, zl = diffusion_map.transition_mat(kernel_matrix, return_z=True)
    e, v = jnp.linalg.eigh(matrix_a)
    # return e, jnp.matmul(z_mat, v), kernel_matrix
    return e, v, zl, kernel_matrix
  else:
    matrix_a = diffusion_map.transition_mat(kernel_matrix)
    return jnp.linalg.eigvalsh(matrix_a)

#@title diff S functoons
def compute_diffusion_map_eigens_reuseS(
    similarity_mat_dict, 
    epsilon_array,  # list of epsilon values for 
):
  """Returns a mapping from `similarity_mat_dict.keys()` to evals for all epsilon.""" 
  output_vals = {}
  for key, similarity_mat in tqdm(similarity_mat_dict.items()):
    _diffusion_map_eigens_fn = jax.vmap(_diffusion_map_eigens, in_axes=(None, 0))
    _diffusion_map_eigens_fn_fit = jax.jit(_diffusion_map_eigens_fn)
    output_vals[key] = _diffusion_map_eigens_fn_fit(similarity_mat, epsilon_array)
  return output_vals
  
def compute_diffusion_map_evecs_and_kernels(    
    all_params,  # mapping from keys to stacked parameters;
    small_epsilon_list,  # list of epsilon values for 
    similarity_fn, 
    return_A_evec=False, 
    kernel_dict=None, # Dictionary of kernel matrices
    similarity_post_process_fn=None,  # function that updated similarity matrix
    return_processed=False, 
):
  """Returns a mapping from `all_params.keys() + epsilon` to DM evecs."""
  evec_output_vals = {}
  kernel_output_vals = {}
  similarity_output_vals = {}
  zl_output_vals = {}
  overlap_data_vals = {}
  vector_data_vals = {}
  _diffusion_map_eigens_fn = functools.partial(_diffusion_map_eigens,
                                             vectors_and_kernel=True)
  for key, stacked_params in tqdm(all_params.items()):
    if kernel_dict is not None:
      similarity_mat = kernel_dict[key]
    else:
      similarity_mat = _get_similarity_matrix(similarity_fn, stacked_params)
      if similarity_post_process_fn is not None:
        if return_processed:
          similarity_mat, (overlap_data, vectors_data) = similarity_post_process_fn(
              similarity_mat, stacked_params,
              return_processed=return_processed)
          overlap_data_vals[key] = overlap_data
          vector_data_vals[key] = vectors_data
        else:
          similarity_mat = similarity_post_process_fn(similarity_mat, stacked_params)
        
    similarity_output_vals[key] = similarity_mat

    for eps in small_epsilon_list:
      _, evec, zl, kernel = _diffusion_map_eigens_fn(similarity_mat, eps)
      if return_A_evec:
        evec_rescale = evec
      else:
        evec_rescale = evec @ np.diag(1. / np.sqrt(zl))
      evec_output_vals[key + (eps,)] = evec_rescale
      kernel_output_vals[key + (eps,)] = kernel
      zl_output_vals[key + (eps,)] = zl
  if return_processed:
    return evec_output_vals, similarity_output_vals, kernel_output_vals, zl_output_vals, overlap_data_vals, vector_data_vals
  return evec_output_vals, similarity_output_vals, kernel_output_vals, zl_output_vals

def estimate_w_loops(new_samples, psi_apply, rng_seq, spin_shape, len_chain_E, burn_E_factor, num_samples_E, file_path, angle):
  # Parameters for mcmc energy
  num_spins = spin_shape[0] * spin_shape[1]
  len_chain_burn_E = burn_E_factor * num_spins
  directions = [0, 1]
  loop_indices = range(spin_shape[1])
  file_path_mcmc = file_path + "MCMC_estimations/"
  results_desc = ["MCMC_ev", "MCMC_std", "MCMC_local_vals", "MCMC_accept_mean"]
  op_dict_desc = "WL_H"
  for key, data in new_samples.items():
    T, h_field = key  #Decode key to parameters
    data_size = jax.tree_leaves(utils.shape_structure(data))[0]
    file_name = f"_{op_dict_desc}_{spin_shape}_hz{h_field}_T{T}.p"
    # Build operator dictionary
    op_dict = {}
    ham = tc_utils.set_up_ham_field_rotated(spin_shape, h_field, angle) # Build hamiltonian operator
    op_dict['ham'] = ham
    for dir in directions:  # Build sets of wilson loop operators
      for idx in loop_indices:
        label = [f'WLX{idx}', f'WLY{idx}']
        loop_x = operators.WilsonLXBond(bond=bonds.wilson_loops(spin_shape, dir, idx))
        op_dict[label[dir]] = loop_x
    # Define the jitted estimation function for a set of operators
    estimate_op_dict_fn = functools.partial(estimates_mcmc.estimate_operator_dict, operator_dict=op_dict, psi_apply=psi_apply, 
                                    len_chain=len_chain_E, len_chain_first_burn=len_chain_burn_E, spin_shape=spin_shape, 
                                    num_samples=num_samples_E)
    estimate_op_dict_jit = jax.jit(estimate_op_dict_fn)
    # MCMC estimates
    rng_keys = utils.split_key(next(rng_seq), (data_size, 2))
    results1_list = []
    results2_list = []
    results3_list = []
    results4_list = []
    for i in tqdm(range(data_size)):
      param = utils.slice_along_axis(data, axis=0, idx=i)
      results = estimate_op_dict_jit(rng_keys[i, ...], param)
      results1_list.append(results[0])
      results2_list.append(results[1])
      results3_list.append(results[2])
      results4_list.append(results[3])
    # results = estimate_op_dict_jit(rng_keys, data)
    results_final_list = [utils.stack_along_axis(results1_list, 0), 
                     utils.stack_along_axis(results2_list, 0), 
                     utils.stack_along_axis(results3_list, 0), 
                     utils.stack_along_axis(results4_list, 0)]
    results_final_dict = dict(zip(results_desc, results_final_list))                  
    for i, results in enumerate(results_final_list):
      file_name_result = results_desc[i]+ file_name
      save_file(results, file_name_result, file_path_mcmc)  

# def avg_loop(all_mcmc_estimation_dict, mcmc_dict_desc="MCMC_ev", op_desc=["WLX", "WLY"]):
#   wloop_absmax_dict = {}
#   data_dict = {k: v[mcmc_dict_desc] for k, v in all_mcmc_estimation_dict.items()}
#   output_dict = {}
#   for key, value in data_dict.items():
#     T, h_field = key
#     values_desc_list = []
#     for desc in op_desc:
#       values = np.stack([v for k, v in value.items() if k.startswith(desc)])
#       # argmax_indices = np.argmax(abs(values), 0)
#       my_values = np.mean(values, 0)
#       values_desc_list.append(my_values)
#     output_dict[key] = np.stack(values_desc_list).T
#   return output_dict     

def process_loop(all_mcmc_estimation_dict, mode, mcmc_dict_desc="MCMC_ev", op_desc=["WLX", "WLY"]):
  wloop_absmax_dict = {}
  data_dict = {k: v[mcmc_dict_desc] for k, v in all_mcmc_estimation_dict.items()}
  output_dict = {}
  for key, value in data_dict.items():
    T, h_field = key
    values_desc_list = []
    for desc in op_desc:
      values = np.stack([v for k, v in value.items() if k.startswith(desc)])
      if mode == 'stack': # todo: deprecate this. 
        values_processed = values
      elif mode == 'avg':
        values_processed = np.mean(values, 0, keepdims=False)
      elif mode == 'abs max': # todo: remove the extra dimension
        argmax_indices = np.argmax(abs(values), 0)
        values_processed = np.take_along_axis(values, argmax_indices[np.newaxis, ...], axis=0)
        # values_processed = np.squeeze(my_values, 0)

      else:
        raise f"Specified mode {mode} is not found."
      values_desc_list.append(values_processed)
    output_dict[key] = np.stack(values_desc_list).T
  return output_dict     

def plot_w_loop(fig, axes, w_dict):
  rows_labels, cols_labels = zip(*w_dict.keys()) 
  rows_labels = sorted(set(rows_labels))
  cols_labels = sorted(set(cols_labels))
  for i, T in enumerate(rows_labels):
    for j, h_field in enumerate(cols_labels):
      data = w_dict[(T, h_field)]
      ax = axes[i, j]
      data_x = data[..., 0]
      data_y = data[..., 1]
      ax.scatter(np.arange(len(data_x)),data_x, label=f"<$W_x$>", s=1)
      ax.scatter(np.arange(len(data_y)),data_y, label=f"<$W_y$>", s=1)
      ax.legend(bbox_to_anchor=(1.0, 1.), loc='upper left')
      ax.set_xlabel("$\Lambda_{l}$", fontsize=12)     
      ax.set_ylabel("$W_{1, 2}$", fontsize=12)
      ticks = np.arange(0, total_samples+1, len_chain)
      ax.set_xticks(ticks)    
      ax.vlines(x=ticks, ymin=-1, ymax=1, colors='purple', ls='--', lw=2)
  pad = 5       
  for ax, row in zip(axes[:, 0], rows_labels):
    ax.annotate(f"$T=${row}", xy=(0., 0.5), xytext=(-15 * pad, 0),
                  xycoords='axes fraction', textcoords='offset points',
                  size='large', ha='center', va='center', fontsize=15, rotation=90)     
  for ax, col in zip(axes[0], cols_labels):
    ax.annotate(f"$h=${col}", xy=(0.5, 1.1), xytext=(0, pad),
                  xycoords='axes fraction', textcoords='offset points',
                  size='large', ha='center', va='baseline', fontsize=15)   