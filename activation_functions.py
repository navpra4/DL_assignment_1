#activation functions
import numpy as np
def sigmoid(z):
  z = np.clip(z,-200,200)
  return 1/(1+np.exp(-z))

def relu(z):
  return np.maximum(0,z)

def tanh(z):
  z = np.clip(z,-200,200)
  return np.tanh(z)

def softmax(z):
  z = np.clip(z,-200,200)
  exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
  return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def det_sigmoid(z):
  sig = sigmoid(z)
  return sig*(1-sig)

def det_tanh(z):
  dtanh = tanh(z)
  return 1-(dtanh*dtanh)

def det_relu(z):
  return np.where(z>=0,1,0)

def identity(z):
  return z

def det_identity(z):
  return np.ones_like(z)
