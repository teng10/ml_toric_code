#@title optimization with MCMC sampling
import jax
import functools
import jax.numpy as jnp
import optax
import train_utils
import utils

def MCMC_optimization(key, batched_configs, batched_psis, psi, model_params, opt_update, opt_state, num_steps, 
                      len_chain, propose_move_fn, make_move_fn, ham, learning_rate, p, epsilon_inv=0.001):
  def _MCMC_step(carry, inputs):
    """
    Return new_chains, new_psis, num_accepts, new_model_params, energy, for a single step after walking 'len_chain' 
    """
    carried_configs, carried_psis, carried_model_params, carried_opt_state = carry     #Define carries
    rngs = inputs     #rngs for all equilibrated chains in the batch for a single step

    update_chain_fn = functools.partial(update_chain, 
                                        len_chain=len_chain, psi=psi, 
                                        propose_move_fn=propose_move_fn, make_move_fn=make_move_fn, 
                                        ham=ham, p=p)
    vectorized_update_chain = jax.vmap(update_chain_fn, in_axes=(0, 0, 0, None))
    new_batch_configs, new_batch_psis, num_accepts = vectorized_update_chain(rngs, carried_configs, carried_psis, carried_model_params)      #Equilibrate chains
    (grad_energy_expectation, energy_expectation, grad_psi_expectation, grad_psi_batch) = train_utils.grad_energy_expectation_gradbatch_fn(
                                                                    batched_configs, batched_psi, 
                                                                    psi, model_params, ham) 
    # print(f" grad_energy_expecatation  {jax.tree_map(lambda x: x, grad_energy_expectation)}")
    # Transform the gradients using the optimiser.
    updates, opt_state = opt_update(grad_energy_expectation, carried_opt_state, carried_model_params)
    # Update parameters.
    new_model_params = optax.apply_updates(carried_model_params, updates)

    # Update psis after changing parameters
    psi_vectorized = jax.vmap(psi, in_axes=(None, 0))
    new_batch_psis_updated = psi_vectorized(new_model_params, new_batch_configs)

    return ((new_batch_configs, new_batch_psis_updated, new_model_params, opt_state), 
      (num_accepts, energy_expectation, grad_psi_expectation, grad_energy_expectation))

  num_batch = batched_configs.shape[0]      #batch size
  rngs = utils.split_key(key, jnp.array([num_steps, num_batch, 2]))

  ((new_batch_configs, new_batch_psis_updated, new_model_params, opt_state), 
      (num_accepts, energy_expectation, grad_psi_expectation, grad_energy_expectation)) = jax.lax.scan(f=_MCMC_step, 
      init=(batched_configs, batched_psis, model_params, opt_state),  xs=(rngs))

  return ((new_batch_configs, new_batch_psis, new_model_params, opt_state), 
    (num_accepts, energy_expectation, grad_psi_expectation, grad_energy_expectation))