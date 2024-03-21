import numpy as np

def update_momentum(params,grad_val,no_hl,learning_rate,u,beta,lamda):
  for idx in range(1,no_hl+2):
    u['W'+str(idx)] = beta*u['W'+str(idx)] + grad_val['W'+str(idx)] + 2*lamda*params['W'+str(idx)]
    params['W'+str(idx)] = params['W'+str(idx)]-learning_rate*u['W'+str(idx)]
    u['B'+str(idx)] = beta*u['B'+str(idx)].flatten() + grad_val['B'+str(idx)].flatten() + 2*lamda*params['B'+str(idx)].flatten()
    params['B'+str(idx)] = params['B'+str(idx)].flatten() - learning_rate*(u['B'+str(idx)].flatten())
  return params,u

def update_nestrov(params,grad_val,no_hl,learning_rate,u,beta,lamda):
  for idx in range(1,no_hl+2):
    u['W'+str(idx)] = beta*u['W'+str(idx)] + grad_val['W'+str(idx)]+ 2*lamda*params['W'+str(idx)]
    u['B'+str(idx)] = beta*(u['B'+str(idx)].flatten()) + grad_val['B'+str(idx)].flatten() + 2*lamda*params['B'+str(idx)].flatten()
    params['W'+str(idx)] = params['W'+str(idx)] - learning_rate*(beta*u['W'+str(idx)] + grad_val['W'+str(idx)]+ 2*lamda*params['W'+str(idx)])
    params['B'+str(idx)] = params['B'+str(idx)].flatten() - learning_rate*(beta*u['B'+str(idx)] + grad_val['B'+str(idx)].flatten() + 2*lamda*params['B'+str(idx)].flatten())
  return params,u

def update_rmsprop(params,grad_val,no_hl,learning_rate,u,beta,eps,lamda):
  for idx in range(1,no_hl+2):
    u['W'+str(idx)] = beta*u['W'+str(idx)] + (1-beta)*np.square(grad_val['W'+str(idx)] + 2*lamda*params['W'+str(idx)])
    params['W'+str(idx)] = params['W'+str(idx)]-learning_rate*(grad_val['W'+str(idx)]+2*lamda*params['W'+str(idx)])/(np.sqrt(u['W'+str(idx)])+eps)
    u['B'+str(idx)] = beta*u['B'+str(idx)].flatten() + (1-beta)*np.square(grad_val['B'+str(idx)].flatten() + 2*lamda*params['B'+str(idx)].flatten())
    params['B'+str(idx)] = params['B'+str(idx)].flatten() - learning_rate*(grad_val['B'+str(idx)].flatten()+ 2*lamda*params['B'+str(idx)].flatten())/(np.sqrt(u['B'+str(idx)].flatten())+eps)
  return params,u

def update_adam(params,grad_val,no_hl,learning_rate,u,v,beta1,beta2,eps,its,lamda):
  u_hat={}
  v_hat={}
  for idx in range(1,no_hl+2):
    u['W'+str(idx)] = beta1*u['W'+str(idx)] + (1-beta1)*(grad_val['W'+str(idx)]+2*lamda*params['W'+str(idx)])
    u['B'+str(idx)] = beta1*u['B'+str(idx)].flatten() + (1-beta1)*(grad_val['B'+str(idx)].flatten()+2*lamda*params['B'+str(idx)].flatten())
    v['W'+str(idx)] = beta2*v['W'+str(idx)] + (1-beta2)*np.square(grad_val['W'+str(idx)] + 2*lamda*params['W'+str(idx)])
    v['B'+str(idx)] = beta2*v['B'+str(idx)].flatten() + (1-beta2)*np.square(grad_val['B'+str(idx)].flatten()+2*lamda*params['B'+str(idx)].flatten())

    u_hat['W'+str(idx)] = u['W'+str(idx)]/(1-np.power(beta1,its+1))
    u_hat['B'+str(idx)] = u['B'+str(idx)]/(1-np.power(beta1,its+1))
    v_hat['W'+str(idx)] = v['W'+str(idx)]/(1-np.power(beta2,its+1))
    v_hat['B'+str(idx)] = v['B'+str(idx)]/(1-np.power(beta2,its+1))

    params['W'+str(idx)] = params['W'+str(idx)]-learning_rate*u_hat['W'+str(idx)]/(np.sqrt(v_hat['W'+str(idx)])+eps)
    params['B'+str(idx)] = params['B'+str(idx)].flatten() - learning_rate*(u_hat['B'+str(idx)].flatten())/(np.sqrt(v_hat['B'+str(idx)].flatten())+eps)
  return params,u,v

def update_nadam(params,grad_val,no_hl,learning_rate,u,v,beta1,beta2,eps,its,lamda):
  u_hat={}
  v_hat={}
  for idx in range(1,no_hl+2):
    u['W'+str(idx)] = beta1*u['W'+str(idx)] + (1-beta1)*(grad_val['W'+str(idx)]+2*lamda*params['W'+str(idx)])
    u['B'+str(idx)] = beta1*u['B'+str(idx)].flatten() + (1-beta1)*(grad_val['B'+str(idx)].flatten()+2*lamda*params['B'+str(idx)].flatten())
    v['W'+str(idx)] = beta2*v['W'+str(idx)] + (1-beta2)*np.square(grad_val['W'+str(idx)]+2*lamda*params['W'+str(idx)])
    v['B'+str(idx)] = beta2*v['B'+str(idx)].flatten() + (1-beta2)*np.square(grad_val['B'+str(idx)].flatten()+2*lamda*params['B'+str(idx)].flatten())

    u_hat['W'+str(idx)] = u['W'+str(idx)]/(1-np.power(beta1,its+1))
    u_hat['B'+str(idx)] = u['B'+str(idx)]/(1-np.power(beta1,its+1))
    v_hat['W'+str(idx)] = v['W'+str(idx)]/(1-np.power(beta2,its+1))
    v_hat['B'+str(idx)] = v['B'+str(idx)]/(1-np.power(beta2,its+1))
    num = beta1*u_hat['W'+str(idx)] + (1-beta1)*(grad_val['W'+str(idx)]+2*lamda*params['W'+str(idx)])/(1-np.power(beta1,its+1))
    params['W'+str(idx)] = params['W'+str(idx)]-learning_rate*(num)/(np.sqrt(v_hat['W'+str(idx)])+eps)
    num = beta1*u_hat['B'+str(idx)] + (1-beta1)*(grad_val['B'+str(idx)].flatten() + 2*lamda*params['B'+str(idx)].flatten())/(1-np.power(beta1,its+1))
    params['B'+str(idx)] = params['B'+str(idx)].flatten() - learning_rate*(num)/(np.sqrt(v_hat['B'+str(idx)].flatten())+eps)
  return params,u,v

def update(params,grad_val,no_hl,learning_rate,lamda):
  for idx in range(1,no_hl+2):
    params['W'+str(idx)] = params['W'+str(idx)]-learning_rate*(grad_val['W'+str(idx)]+2*lamda*params['W'+str(idx)])
    params['B'+str(idx)] = params['B'+str(idx)].flatten() - learning_rate*(grad_val['B'+str(idx)].flatten() + 2*lamda*params['B'+str(idx)].flatten())
  return params