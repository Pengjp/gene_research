import numpy as np
import tensorflow as tf

@tf.function
def inverse(x, R_in):
  ''' Require tensorflow '''
  ''' Input: 
	x: a real number
	R: a neuron netweek with one hidden layer and RELU activation 
       Return:
	It computes the inverst of R_in by regarding R_in as a piecewise linear continuous function
	Example of R_in (built in keras):
	
	R_in = keras.Sequential(
    [
        keras.Input(shape=(1,)),
        layers.Dense(H, activation="relu", name="hidden"),
        layers.Dense(1, activation=None, name="output"),
    ]
)
  '''
  x = tf.dtypes.cast(x, tf.float32)
  w, b = R_in.get_layer('hidden').weights
  v, v0 = R_in.get_layer('output').weights

  num_layer = w.shape[1]

  b_over_w = - b / w # the negative sign is important
  b_over_w = tf.reshape(b_over_w, [-1])# change it to row vector
  # Then we need to compute range for the inverse function
  r_x = tf.reshape(
      [tf.matmul( ( tf.nn.relu(w * b_over_w[i] + b) ), v) + v0 for i in range(1)], 
      [-1]
      )
  # pad r_x with 0 and inf
  # right side gives us the right index
  pos = tf.searchsorted(tf.sort(tf.concat([[0], r_x, [np.inf]],0)), x,side='right') # get the interval index
  # tf.print('pos',pos)
  pos = tf.reshape(pos,[]) # get only numerical value
  # change it to row vector
  v = tf.reshape(v, [-1])
  w = tf.reshape(w, [-1])
  
  index = tf.argsort(r_x)
  # tf.print(pos,"\n")
  
  # sort according to the index
  v_b = tf.gather(v * b, index)
  v_w = tf.gather(v * w, index)

  num = x - v0 - tf.reduce_sum(v_b[:(pos - 1)])
  deo = tf.reduce_sum(v_w[:(pos - 1)])
  
  return tf.math.divide_no_nan(num, deo)[0]



def runif():
    return tf.random.uniform([1], dtype=tf.float64)[0]

def rexp():
    return -tf.math.log(runif())

@tf.function
def gen_gaps(R_in, k: int,  
             theta=tf.constant(1e-4, dtype=tf.float64), 
             rho=tf.constant(1e-5, dtype=tf.float64)) -> tf.Tensor:
    '''Return k gaps sampled from genetic distribution with rate function eta.'''
    z = tf.convert_to_tensor([[rexp()]])
    x = tf.dtypes.cast(inverse(z, R_in), dtype=tf.float64)[0]  # initialize x by sampling from prior
    pos = tf.constant(0., dtype=tf.float64)
    j = 0
    ta = tf.TensorArray(tf.float64, size=k + 2)
    while tf.less(j, k + 2):
        # x' satisfies R(x') - R(u*x) = Z => x' = Rinv(Z + R(u*x))
        u = runif()
        z = rexp()
        u_x = tf.convert_to_tensor([[u * x]])
        r_u_x = tf.dtypes.cast(R_in(u_x), dtype=tf.float64)
        x = tf.dtypes.cast((inverse((z + r_u_x), R_in) ),dtype=tf.float64 )[0]  # segment height
        with tf.control_dependencies(
            [tf.debugging.assert_positive(x)]
        ):
          pos += rexp() / (x * (theta + rho))  # length to next event
        while runif() < (theta / (theta + rho)) and tf.less(j, k + 2):
            ta = ta.write(j, pos)
            j += 1
            pos += rexp() / (x * (theta + rho))  # length to next event
    ret = ta.stack()[1:]  # first obs suffers from inspection paradox?
    diff = ret[1:] - ret[:-1]

    with tf.control_dependencies([
        tf.debugging.assert_positive(diff)
    ]):
        return tf.cast(diff, tf.float32)