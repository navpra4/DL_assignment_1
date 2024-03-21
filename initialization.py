import numpy as np

def init_layers(no_hl,size_hl):
  params = {}
  params['W1'] = np.random.uniform(low=-1,high=1,size=(size_hl,784))
  params['B1'] = np.random.uniform(low=-1,high=1,size=(size_hl,1))
  for idx in range(2,no_hl+1):
    params['W'+str(idx)] = np.random.uniform(low=-1,high=1,size=(size_hl,size_hl))
    params['B'+str(idx)] = np.random.uniform(low=-1,high=1,size=(size_hl,1))
  params['W'+str(no_hl+1)] = np.random.uniform(low=-1,high=1,size=(10,size_hl))
  params['B'+str(no_hl+1)] = np.random.uniform(low=-1,high=1,size=(10,1))
  return params

def init_layers_xavier(no_hl,size_hl):
  params = {}
  params['W1'] = np.random.uniform(low=-np.sqrt(6/(size_hl+784)),high=np.sqrt(6/(size_hl+784)),size=(size_hl,784))
  params['B1'] = np.random.uniform(low=-np.sqrt(6/(size_hl+784)),high=np.sqrt(6/(size_hl+784)),size=(size_hl,1))
  for idx in range(2,no_hl+1):
    params['W'+str(idx)] = np.random.uniform(low=-np.sqrt(6/(size_hl*2)),high=np.sqrt(6/(size_hl*2)),size=(size_hl,size_hl))
    params['B'+str(idx)] = np.random.uniform(low=-np.sqrt(6/(size_hl+1)),high=np.sqrt(6/(size_hl+1)),size=(size_hl,1))
  params['W'+str(no_hl+1)] = np.random.uniform(low=-np.sqrt(6/(size_hl+10)),high=np.sqrt(6/(size_hl+10)),size=(10,size_hl))
  params['B'+str(no_hl+1)] = np.random.uniform(low=-np.sqrt(6/(11)),high=np.sqrt(6/(11)),size=(10,1))
  return params

def init_u(no_hl,size_hl):
  u = {}
  u['W1'] = np.zeros((size_hl,784))
  u['B1'] = np.zeros((size_hl,1))
  for idx in range(2,no_hl+1):
    u['W'+str(idx)] = np.zeros((size_hl,size_hl))
    u['B'+str(idx)] = np.zeros((size_hl,1))
  u['W'+str(no_hl+1)] = np.zeros((10,size_hl))
  u['B'+str(no_hl+1)] = np.zeros((10,1))
  return u