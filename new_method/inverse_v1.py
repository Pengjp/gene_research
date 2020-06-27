#inverse.py

def inverse(x,R_in):
  w,b=R.get_layer('hidden').get_weights()
  w=w.reshape([-1])
  v,v0=R.get_layer('output').get_weights()
  v=v.reshape([-1])
  # Create a dataframe to hold all parameters
  df=df=pd.DataFrame(data=np.array([w,b,v]).T, columns=['w','b','v'], dtype=np.float64)
  df['-b/w'] = - b / w
  df['v*b'] = df.v * df.b
  df['v*w'] = df.v * df.w
  df['r_x'] = np.zeros([10,1], dtype=np.float64)
  for i in range(len(df)):
    df.at[i, 'r_x']= np.array(tf.keras.backend.eval(R(np.array([[df.loc[i]['-b/w']]])))[0], dtype=np.float32)
  df.sort_values(['r_x'], inplace=True)
  df.reset_index(drop=True, inplace=True)
  # find the correct interval
  pos=np.searchsorted(df['r_x'],x) # this returns 
  if x <= v0: # consant function part
    return min(df['-b/w'])
  if pos == len(df): # rightmost part
    return (x - v0 - sum(df['v*b'])) / sum(df['v*w'])
  else:
    return (x - v0 - sum(df.loc[:(pos - 1)]['v*b']) ) / (sum(df.loc[:(pos - 1)]['v*w']))
