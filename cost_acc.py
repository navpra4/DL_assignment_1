import numpy as np

def get_cost(Y_hat,Y,loss):
  m = Y_hat.shape[0]
  if loss=="cross_entropy":
    err = np.array([Y_hat[i][Y[i]] for i in range(m)])
    return (np.sum(-1*np.log(err)))/m
  elif loss=="mean_squared_error":
    yt=np.zeros_like(Y_hat)
    for i in range(Y_hat.shape[0]):
      yt[i][Y[i]]=1
    return np.mean(np.square(yt-Y_hat))

def get_accuracy(Y_hat,Y):
  yhat = np.argmax(Y_hat,axis=1)
  return np.mean(yhat == Y)