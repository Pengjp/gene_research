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
  x = tf.dtypes.cast(x, tf.float32) # make sure input data is float32
  # It returns a tf.Variable
  w, b = R_in.get_layer('hidden').weights
  v, v0 = R_in.get_layer('output').weights

  b_over_w = - b / w # the negative sign is important
  b_over_w = tf.reshape(b_over_w, [-1])# change it to row vector
	
  # Then we need to compute range for the inverse function
  r_x = tf.reshape(
      [tf.matmul( ( tf.nn.relu(w * b_over_w[i] + b) ), v) + v0 for i in range(10)], 
      [-1]
      )
  r_x_sorted = tf.sort(tf.concat([[0], r_x, [np.inf]],0))	
  # get the index of correct interval
  
  pos = tf.searchsorted(r_x_sorted, x,side='right') # get the interval index

  pos = tf.reshape(pos,[]) # get only numerical value
  # change it to row vector
  v = tf.reshape(v, [-1])
  w = tf.reshape(w, [-1])
  
  index = tf.argsort(r_x)
  
  # sort according to the index
  v_b = tf.gather(v * b, index)
  v_w = tf.gather(v * w, index)

  num = x - v0 - tf.reduce_sum(v_b[:(pos - 1)])
  deo = tf.reduce_sum(v_w[:(pos - 1)])
  
  return tf.math.divide_no_nan(num, deo)
