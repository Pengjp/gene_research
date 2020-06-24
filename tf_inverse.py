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

  w, b = R_in.get_layer('hidden').get_weights()
  v, v0 = R_in.get_layer('output').get_weights()

  b_over_w = - b / w # the negative sign is important
  b_over_w = b_over_w.reshape([-1,]) # change it to row vector
  # Then we need to compute range for the inverse function
  r_x = tf.convert_to_tensor(
      np.array(([v0 + tf.nn.relu(w * i + b) @ v for i in b_over_w])).reshape([-1,])
      )

  # pad r_x with 0 and inf
  # right side gives us the right index
  pos = tf.searchsorted(tf.sort(tf.concat([[0], r_x, [np.inf]],0)), [x],side='right') # get the interval index
  pos = tf.reshape(pos,[]) # get only numerical value

  # change it to row vector
  v = v.reshape([-1,]) 
  w = w.reshape([-1,]) 

  index = tf.argsort(r_x)

  v_b = tf.gather(v * b, index)
  v_w = tf.gather(v * w, index)

  num = x - v0 - tf.reduce_sum(v_b[:(pos - 1)])
  deo = tf.reduce_sum(v_w[:(pos - 1)])
  
  return tf.math.divide_no_nan(num, deo)
