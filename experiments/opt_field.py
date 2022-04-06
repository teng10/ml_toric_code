import opt_utils

h_step = 0.1
h_field_array=np.round(np.arange(0, .2, h_step), 2)
angle = 0.
file_path = '/n/home11/yteng/experiments/optimization/'
iterations = 3
epsilon = 0.2
model_name = 'rbm_cnn'


spin_shape = (6,3)
num_spins = spin_shape[0] * spin_shape[1]
burn_in_factor = 600
rng_seq = hk.PRNGSequence(42)
sector = 1
params_list_list = []
energies_list = []
energy_steps_list = []
init_param_list = []
for i in range(iterations): 
  main_key = next(rng_seq)
  params_list, energy, psis, energy_steps, psis_list, num_accepts_list, grad_list, init_param = optimization_field.main_no_carry_angle_flexible(h_field_array=h_field_array, epsilon=epsilon, 
                                                                                                        spin_shape=spin_shape, num_chains=300, num_steps=450, 
                                                          first_burn_len=num_spins*burn_in_factor, len_chain=30, learning_rate=0.005, spin_flip_p=.4, main_key=main_key, 
                                                          angle=angle, model_name=model_name, sector=sector)
  params_list_list.append(params_list)
  energies_list.append(energy)
  energy_steps_list.append(energy_steps)
  init_param_list.append(init_param)
params_list_stacked = utils.stack_along_axis(params_list_list, 0)

now = datetime.datetime.now()
pattern = re.compile(r"-\d\d-\d\d")
mo = pattern.search(str(now))
date = mo.group()[1:]

h_field_list = [utils.round_to_2(h) for h in h_field_array]
field_params_dict = dict(zip(h_field_list, params_list_stacked))
file_name = f"{date}_params_list_stacked_{spin_shape}_{sector}.p"
pickle.dump(field_params_dict, open(file_name, 'wb'))
