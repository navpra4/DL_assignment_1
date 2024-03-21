from tqdm import tqdm
import wandb
from initialization import init_layers, init_layers_xavier, init_u
from updates import update_momentum, update_nestrov, update_rmsprop, update_adam, update_nadam, update
from nn_func import feedforward, backward
from cost_acc import get_cost, get_accuracy

def train_sgd(X,Y,x_test,y_test,no_hl,size_hl,epochs,learning_rate,batchsize,lamda,act_func,init_func,loss,beta,beta1,beta2,eps,momentum):
  x_valid = X.reshape(60000,784)[0:6000]
  y_valid = Y[0:6000]
  x_training = X.reshape(60000,784)[6000:]
  y_training = Y[6000:]
  x_test = x_test.reshape(10000,784)
  params = init_func(no_hl,size_hl)
  cost_history = []
  accuracy_history = []
  for i in range(epochs):
    for j in tqdm(range(x_training.shape[0]//batchsize)):
      Y_hat, act_val = feedforward(x_training[j*batchsize:(j+1)*batchsize],params,no_hl,size_hl,act_func)
      grad_val = backward(Y_hat,y_training[j*batchsize:(j+1)*batchsize],params,act_val,no_hl,size_hl,act_func,loss)
      params = update(params,grad_val,no_hl,learning_rate,lamda)
    y_pred_train,_ = feedforward(x_training,params,no_hl,size_hl,act_func)
    y_pred_test,_ = feedforward(x_test,params,no_hl,size_hl,act_func)
    y_pred_valid,_ = feedforward(x_valid,params,no_hl,size_hl,act_func)
    cost_train = get_cost(y_pred_train, y_training,loss)
    cost_test = get_cost(y_pred_test, y_test,loss)
    cost_valid = get_cost(y_pred_valid, y_valid,loss)
    cost_history.append(cost_test)
    accuracy_train = get_accuracy(y_pred_train, y_training)
    accuracy_test = get_accuracy(y_pred_test, y_test)
    accuracy_valid = get_accuracy(y_pred_valid, y_valid)
    accuracy_history.append(accuracy_test)
    wandb.log({
      'epoch':i+1,
      'Training_Accuracy':accuracy_train,
      'Training_loss':cost_train,
      'Validation_Accuracy':accuracy_valid,
      'Validation_loss':cost_valid
      })
  return params, cost_history,accuracy_history

def train_momentum(X,Y,x_test,y_test,no_hl,size_hl,epochs,learning_rate,batchsize,lamda,act_func,init_func,loss,beta,beta1,beta2,eps,momentum):
  x_valid = X.reshape(60000,784)[0:6000]
  y_valid = Y[0:6000]
  x_training = X.reshape(60000,784)[6000:]
  y_training = Y[6000:]
  x_test = x_test.reshape(10000,784)
  params = init_func(no_hl,size_hl)
  cost_history = []
  accuracy_history = []
  u = init_u(no_hl,size_hl)
  for i in range(epochs):
    for j in tqdm(range(x_training.shape[0]//batchsize)):
      Y_hat, act_val = feedforward(x_training[j*batchsize:(j+1)*batchsize],params,no_hl,size_hl,act_func)
      grad_val = backward(Y_hat,y_training[j*batchsize:(j+1)*batchsize],params,act_val,no_hl,size_hl,act_func,loss)
      params,u = update_momentum(params,grad_val,no_hl,learning_rate,u,momentum,lamda)
    y_pred_train,_ = feedforward(x_training,params,no_hl,size_hl,act_func)
    y_pred_test,_ = feedforward(x_test,params,no_hl,size_hl,act_func)
    y_pred_valid,_ = feedforward(x_valid,params,no_hl,size_hl,act_func)
    cost_train = get_cost(y_pred_train, y_training,loss)
    cost_test = get_cost(y_pred_test, y_test,loss)
    cost_valid = get_cost(y_pred_valid, y_valid,loss)
    cost_history.append(cost_test)
    accuracy_train = get_accuracy(y_pred_train, y_training)
    accuracy_test = get_accuracy(y_pred_test, y_test)
    accuracy_valid = get_accuracy(y_pred_valid, y_valid)
    accuracy_history.append(accuracy_test)
    wandb.log({
      'epoch':i+1,
      'Training_Accuracy':accuracy_train,
      'Training_loss':cost_train,
      'Validation_Accuracy':accuracy_valid,
      'Validation_loss':cost_valid
      })
  return params, cost_history,accuracy_history

def train_nestrov(X,Y,x_test,y_test,no_hl,size_hl,epochs,learning_rate,batchsize,lamda,act_func,init_func,loss,beta,beta1,beta2,eps,momentum):
  x_valid = X.reshape(60000,784)[0:6000]
  y_valid = Y[0:6000]
  x_training = X.reshape(60000,784)[6000:]
  y_training = Y[6000:]
  x_test = x_test.reshape(10000,784)
  params = init_func(no_hl,size_hl)
  cost_history = []
  accuracy_history = []
  u = init_u(no_hl,size_hl)
  for i in range(epochs):
    for j in tqdm(range(x_training.shape[0]//batchsize)):
      Y_hat, act_val = feedforward(x_training[j*batchsize:(j+1)*batchsize],params,no_hl,size_hl,act_func)
      grad_val = backward(Y_hat,y_training[j*batchsize:(j+1)*batchsize],params,act_val,no_hl,size_hl,act_func,loss)
      params,u = update_nestrov(params,grad_val,no_hl,learning_rate,u,momentum,lamda)
    y_pred_train,_ = feedforward(x_training,params,no_hl,size_hl,act_func)
    y_pred_test,_ = feedforward(x_test,params,no_hl,size_hl,act_func)
    y_pred_valid,_ = feedforward(x_valid,params,no_hl,size_hl,act_func)
    cost_train = get_cost(y_pred_train, y_training,loss)
    cost_test = get_cost(y_pred_test, y_test,loss)
    cost_valid = get_cost(y_pred_valid, y_valid,loss)
    cost_history.append(cost_test)
    accuracy_train = get_accuracy(y_pred_train, y_training)
    accuracy_test = get_accuracy(y_pred_test, y_test)
    accuracy_valid = get_accuracy(y_pred_valid, y_valid)
    accuracy_history.append(accuracy_test)
    wandb.log({
      'epoch':i+1,
      'Training_Accuracy':accuracy_train,
      'Training_loss':cost_train,
      'Validation_Accuracy':accuracy_valid,
      'Validation_loss':cost_valid
      })
  return params, cost_history,accuracy_history

def train_rmsprop(X,Y,x_test,y_test,no_hl,size_hl,epochs,learning_rate,batchsize,lamda,act_func,init_func,loss,beta,beta1,beta2,eps,momentum):
  x_valid = X.reshape(60000,784)[0:6000]
  y_valid = Y[0:6000]
  x_training = X.reshape(60000,784)[6000:]
  y_training = Y[6000:]
  x_test = x_test.reshape(10000,784)
  params = init_func(no_hl,size_hl)
  cost_history = []
  accuracy_history = []
  u = init_u(no_hl,size_hl)
  for i in range(epochs):
    for j in tqdm(range(x_training.shape[0]//batchsize)):
      Y_hat, act_val = feedforward(x_training[j*batchsize:(j+1)*batchsize],params,no_hl,size_hl,act_func)
      grad_val = backward(Y_hat,y_training[j*batchsize:(j+1)*batchsize],params,act_val,no_hl,size_hl,act_func,loss)
      params,u = update_rmsprop(params,grad_val,no_hl,learning_rate,u,beta,eps,lamda)
    y_pred_train,_ = feedforward(x_training,params,no_hl,size_hl,act_func)
    y_pred_test,_ = feedforward(x_test,params,no_hl,size_hl,act_func)
    y_pred_valid,_ = feedforward(x_valid,params,no_hl,size_hl,act_func)
    cost_train = get_cost(y_pred_train, y_training,loss)
    cost_test = get_cost(y_pred_test, y_test,loss)
    cost_valid = get_cost(y_pred_valid, y_valid,loss)
    cost_history.append(cost_test)
    accuracy_train = get_accuracy(y_pred_train, y_training)
    accuracy_test = get_accuracy(y_pred_test, y_test)
    accuracy_valid = get_accuracy(y_pred_valid, y_valid)
    accuracy_history.append(accuracy_test)
    wandb.log({
      'epoch':i+1,
      'Training_Accuracy':accuracy_train,
      'Training_loss':cost_train,
      'Validation_Accuracy':accuracy_valid,
      'Validation_loss':cost_valid
      })
  return params, cost_history,accuracy_history

def train_adam(X,Y,x_test,y_test,no_hl,size_hl,epochs,learning_rate,batchsize,lamda,act_func,init_func,loss,beta,beta1,beta2,eps,momentum):
  x_valid = X.reshape(60000,784)[0:6000]
  y_valid = Y[0:6000]
  x_training = X.reshape(60000,784)[6000:]
  y_training = Y[6000:]
  x_test = x_test.reshape(10000,784)
  params = init_func(no_hl,size_hl)
  cost_history = []
  accuracy_history = []
  u = init_u(no_hl,size_hl)
  v = init_u(no_hl,size_hl)
  its=0
  for i in range(epochs):
    for j in tqdm(range(x_training.shape[0]//batchsize)):
      Y_hat, act_val = feedforward(x_training[j*batchsize:(j+1)*batchsize],params,no_hl,size_hl,act_func)
      grad_val = backward(Y_hat,y_training[j*batchsize:(j+1)*batchsize],params,act_val,no_hl,size_hl,act_func,loss)
      params,u,v = update_adam(params,grad_val,no_hl,learning_rate,u,v,beta1,beta2,eps,its,lamda)
      its=its+1
    y_pred_train,_ = feedforward(x_training,params,no_hl,size_hl,act_func)
    y_pred_test,_ = feedforward(x_test,params,no_hl,size_hl,act_func)
    y_pred_valid,_ = feedforward(x_valid,params,no_hl,size_hl,act_func)
    cost_train = get_cost(y_pred_train, y_training,loss)
    cost_test = get_cost(y_pred_test, y_test,loss)
    cost_valid = get_cost(y_pred_valid, y_valid,loss)
    cost_history.append(cost_test)
    accuracy_train = get_accuracy(y_pred_train, y_training)
    accuracy_test = get_accuracy(y_pred_test, y_test)
    accuracy_valid = get_accuracy(y_pred_valid, y_valid)
    accuracy_history.append(accuracy_test)
    wandb.log({
      'epoch':i+1,
      'Training_Accuracy':accuracy_train,
      'Training_loss':cost_train,
      'Validation_Accuracy':accuracy_valid,
      'Validation_loss':cost_valid
      })
  return params, cost_history,accuracy_history

def train_nadam(X,Y,x_test,y_test,no_hl,size_hl,epochs,learning_rate,batchsize,lamda,act_func,init_func,loss,beta,beta1,beta2,eps,momentum):
  x_valid = X.reshape(60000,784)[0:6000]
  y_valid = Y[0:6000]
  x_training = X.reshape(60000,784)[6000:]
  y_training = Y[6000:]
  x_test = x_test.reshape(10000,784)
  params = init_func(no_hl,size_hl)
  cost_history = []
  accuracy_history = []
  u = init_u(no_hl,size_hl)
  v = init_u(no_hl,size_hl)
  its=0
  for i in range(epochs):
    for j in tqdm(range(x_training.shape[0]//batchsize)):
      Y_hat, act_val = feedforward(x_training[j*batchsize:(j+1)*batchsize],params,no_hl,size_hl,act_func)
      grad_val = backward(Y_hat,y_training[j*batchsize:(j+1)*batchsize],params,act_val,no_hl,size_hl,act_func,loss)
      params,u,v = update_nadam(params,grad_val,no_hl,learning_rate,u,v,beta1,beta2,eps,its,lamda)
      its=its+1
    y_pred_train,_ = feedforward(x_training,params,no_hl,size_hl,act_func)
    y_pred_test,_ = feedforward(x_test,params,no_hl,size_hl,act_func)
    y_pred_valid,_ = feedforward(x_valid,params,no_hl,size_hl,act_func)
    cost_train = get_cost(y_pred_train, y_training,loss)
    cost_test = get_cost(y_pred_test, y_test,loss)
    cost_valid = get_cost(y_pred_valid, y_valid,loss)
    cost_history.append(cost_test)
    accuracy_train = get_accuracy(y_pred_train, y_training)
    accuracy_test = get_accuracy(y_pred_test, y_test)
    accuracy_valid = get_accuracy(y_pred_valid, y_valid)
    accuracy_history.append(accuracy_valid)
    wandb.log({
      'epoch':i+1,
      'Training_Accuracy':accuracy_train,
      'Training_loss':cost_train,
      'Validation_Accuracy':accuracy_valid,
      'Validation_loss':cost_valid
    })
  return params, cost_history,accuracy_history

