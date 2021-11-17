#@title optimization with MCMC sampling
import jax
import functools
import jax.numpy as jnp
import optax
import train_utils
import utils
import mcmc
import numpy as np

def MCMC_optimization(key, init_batched_cs, init_batched_psis, psi, init_params, opt_update, init_opt_state, num_steps, 
                      len_chain, propose_move_fn, make_move_fn, ham, learning_rate):
  def _MCMC_step(carry, inputs):
    """
    Return new_chains, new_psis, num_accepts, new_model_params, energy, for a single step after walking 'len_chain' 
    """
    carried_configs, carried_psis, carried_model_params, carried_opt_state = carry     #Define carries
    rngs = inputs     #rngs for all equilibrated chains in the batch for a single step

    update_chain_fn = functools.partial(mcmc.update_chain, 
                                        len_chain=len_chain, psi=psi, 
                                        propose_move_fn=propose_move_fn, make_move_fn=make_move_fn,)
    update_chain_vectorized = jax.vmap(update_chain_fn, in_axes=(0, 0, 0, None))
    new_batch_configs, new_batch_psis, num_accepts = update_chain_vectorized(rngs, 
                                                      carried_configs, carried_psis, carried_model_params)      #Equilibrate chains
    (grad_energy_expectation, energy_expectation, grad_psi_expectation, grad_psi_batch) = train_utils.grad_energy_expectation_gradbatch_fn(
                                                                    new_batch_configs, new_batch_psis, 
                                                                    psi, carried_model_params, ham) 
    # Transform the gradients using the optimiser.
    updates, opt_state = opt_update(grad_energy_expectation, carried_opt_state, carried_model_params)
    # Update parameters.
    new_model_params = optax.apply_updates(carried_model_params, updates)
    # Update psis after changing parameters
    psi_vectorized = jax.vmap(psi, in_axes=(None, 0))
    new_batch_psis_updated = psi_vectorized(new_model_params, new_batch_configs)

    return ((new_batch_configs, new_batch_psis_updated, new_model_params, opt_state), 
      (num_accepts, energy_expectation, grad_psi_expectation, grad_energy_expectation))

  rngs = utils.split_key(key, np.array([num_steps, init_batched_cs.shape[0], 2], dtype=int))
  # Scan bathces for num_steps
  return jax.lax.scan(f=_MCMC_step, 
      init=(init_batched_cs, init_batched_psis, init_params, init_opt_state),  xs=(rngs))