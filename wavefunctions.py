#@title Neural Network Model
import jax.numpy as jnp
import haiku as hk
import einops
import tc_utils

class RBM(hk.Module):

  def __init__(self, spin_shape, name=None):
    super().__init__(name=None)
    self.spin_shape = spin_shape

  def __call__(self, x):
    # size = x.shape[0]
    x_2d = einops.rearrange(x, '(x y)-> x y', x=self.spin_shape[0], y=self.spin_shape[1])
    x_facebond, x_vertexbond = tc_utils.stack_F_V_img(x_2d)
    # print(x_vertexbond[:, 0,0])
    num_nb, shape_x, shape_y = x_facebond.shape
    assert num_nb==4
    wF = hk.get_parameter("wF", shape=[num_nb], dtype=x_2d.dtype, init=jnp.ones)
    bF = hk.get_parameter("bF", shape=[1], dtype=x_2d.dtype, init=jnp.zeros)
    wF = jnp.expand_dims(wF, (1,2))
    wF = jnp.tile(wF, (1, shape_x, shape_y))    
    assert wF.shape == x_facebond.shape, "Weights shape is not the same as spins bonds"
    x_convolved_F = jnp.sum(jnp.multiply(wF, x_facebond), axis=0)
    # assert bF.shape == x_convolved_F.shape, "Bias shape does not match w*sigma shape"
    wV = hk.get_parameter("wV", shape=[num_nb], dtype=x_2d.dtype, init=jnp.ones)
    wV = jnp.expand_dims(wV, (1,2))
    wV = jnp.tile(wV, (1, shape_x, shape_y))     
    bV = hk.get_parameter("bV", shape=[1], dtype=x_2d.dtype, init=jnp.zeros)
    x_convolved_V = jnp.sum(jnp.multiply(wV, x_vertexbond), axis=0)
    output = jnp.prod(jnp.cos(bF + x_convolved_F)) * jnp.prod(jnp.cos(bV + x_convolved_V))

    return output

def fwd(x, spin_shape):
  return RBM(spin_shape)(x)

#@title Neural Network Model
class RBM_CNN(hk.Module):

  def __init__(self, kernels, name=None):
    super().__init__(name=None)
    output_channel = 1

    # Add mask to make sure CNN only takes values from nearest neighbour
    mask_F = np.array([[1,0],[1,1],[1,0]])
    mask_F = np.expand_dims(mask_F, axis=(-1, -2))
    mask_V = np.array([[0,1],[1,1],[0,1]])
    mask_V = np.expand_dims(mask_V, axis=(-1, -2))

    self.conv_F = hk.Conv2D(output_channel, kernels, name="F",stride=(2,1), padding="VALID", mask=mask_F)
    #Build CNN layer for vertex operators
    self.conv_V = hk.Conv2D(output_channel, kernels, name="V",stride=(2,1), padding="VALID", mask=mask_V) 
  
  def __call__(self, x):
    len_x = x.shape[0]
    # print(len_x)
    x_size = np.sqrt((len_x // 2)).astype(int)
    x_2d = jnp.reshape(x, (2 * x_size,  x_size ))
    wrapped_x = jnp.pad(x_2d, ((0, 2), (0, 1)), mode='wrap')

    wrapped_x = jnp.expand_dims(wrapped_x, -1)      #Wrap input x for periodic boundary condition

    F_conv = self.conv_F(wrapped_x)

    x_2d_rolled = jnp.roll(x_2d, -1, axis=0)

    wrapped_x_2d_rolled = jnp.pad(x_2d_rolled, ((0, 2), (0, 1)), mode='wrap')
    wrapped_x_2d_rolled = jnp.expand_dims(wrapped_x_2d_rolled, -1)  
    V_conv = self.conv_V(wrapped_x_2d_rolled)
    # print(V_conv)
    F_conv_activation = jnp.cos(F_conv)
    # print(F_conv_activation)
    V_conv_activation = jnp.cos(V_conv)

    output = jnp.prod(F_conv_activation) * jnp.prod(V_conv_activation)

    return output

def fwd_cnn(x):
  return RBM_CNN(kernels=(3,2))(x)

#@title Neural Network Model
class RBM_noise(hk.Module):
  """
  RBM noise without translational invariance. 
  """
  def __init__(self, spin_shape, name=None):
    super().__init__(name=None)
    self.spin_shape = spin_shape

  def __call__(self, x):
    x_2d = einops.rearrange(x, '(x y)-> x y', x=self.spin_shape[0], y=self.spin_shape[1])
    x_facebond, x_vertexbond = tc_utils.stack_F_V_img(x_2d)
    # print(x_vertexbond[:, 0,0])
    num_nb, shape_x, shape_y = x_facebond.shape
    assert num_nb==4
    wF = hk.get_parameter("wF", shape=[num_nb, shape_x, shape_y], dtype=x_2d.dtype, init=jnp.ones)
    bF = hk.get_parameter("bF", shape=[1, shape_x, shape_y], dtype=x_2d.dtype, init=jnp.zeros)
    assert wF.shape == x_facebond.shape, "Weights shape is not the same as spins bonds"
    x_convolved_F = jnp.sum(jnp.multiply(wF, x_facebond), axis=0)
    # assert bF.shape == x_convolved_F.shape, "Bias shape does not match w*sigma shape"
    wV = hk.get_parameter("wV", shape=[num_nb, shape_x, shape_y], dtype=x_2d.dtype, init=jnp.ones)
    bV = hk.get_parameter("bV", shape=[1, shape_x, shape_y], dtype=x_2d.dtype, init=jnp.zeros)
    x_convolved_V = jnp.sum(jnp.multiply(wV, x_vertexbond), axis=0)
    output = jnp.prod(jnp.cos(bF + x_convolved_F)) * jnp.prod(jnp.cos(bV + x_convolved_V))

    return output

def fwd_noise(x, spin_shape):
  return RBM_noise(spin_shape)(x)