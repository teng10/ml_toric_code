#@ utils

def split_key(key, new_shape):
  keys_dim = new_shape[:-1]
  # print(keys_dim)
  rng = jax.random.split(key, jnp.prod(keys_dim))      #Split the keys based on batch size and steps
  rngs = jnp.reshape(rng, new_shape)
  return rngs