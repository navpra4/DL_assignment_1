import numpy as np
from activation_functions import sigmoid, relu, tanh, softmax, det_sigmoid, det_tanh, det_relu, identity, det_identity

def feedforward(X, params, no_hl, size_hl, act_func):
    """
    Perform the feedforward pass through the neural network to compute the predicted output.

    Parameters:
    - X (numpy.ndarray): Input data or features.
    - params (dict): Dictionary containing weights and biases of the network.
    - no_hl (int): Number of hidden layers in the neural network.
    - size_hl (int): Size of each hidden layer (not used in this function).
    - act_func (function): Activation function used in the network.

    Returns:
    - Y_hat (numpy.ndarray): Predicted output of the neural network.
    - act_val (dict): Dictionary containing activation values from each layer of the network.
                      Keys are in the format 'H0', 'A1', 'H1', 'A2', ..., 'Hn-1', 'An'.
                      Values are the corresponding activation matrices or vectors.
    """
    act_val = {}
    h_curr = X
    for idx in range(1, no_hl + 2):
        h_prev = h_curr
        w_curr = params['W' + str(idx)]
        b_curr = params['B' + str(idx)]
        a_curr = np.dot(h_prev, w_curr.T) + b_curr.T
        h_curr = act_func(a_curr)
        act_val['H' + str(idx - 1)] = h_prev
        act_val['A' + str(idx)] = a_curr
    return softmax(act_val['A' + str(no_hl + 1)]), act_val

def backward(Y_hat, Y, params, act_val, no_hl, size_hl,act_func,loss):
    """
    Perform backpropagation to compute the gradients of weights and biases.

    Parameters:
    - Y_hat (numpy.ndarray): Predicted output from the feedforward pass.
    - Y (numpy.ndarray): Actual target output.
    - params (dict): Dictionary containing weights and biases of the network.
    - act_val (dict): Dictionary containing activation values from the feedforward pass.
    - no_hl (int): Number of hidden layers in the neural network.
    - size_hl (int): Size of each hidden layer (not used in this function).
    - act_func (function): Activation function used in the network.
    - loss (str): Type of loss function used, either "cross_entropy" or "mean_squared_error".

    Returns:
    - grad_val (dict): Dictionary containing gradients of weights and biases for all layers.
                      Keys are in the format 'W1', 'B1', 'W2', 'B2', ..., 'Wn', 'Bn'.
                      Values are the corresponding gradient matrices or vectors.
    """
    grad_val = {}
    e_y = np.eye(10)[Y]
    if loss=="cross_entropy":
      dA_prev = -(e_y - Y_hat)
    elif loss =="mean_squared_error":
      dA_prev = (Y_hat - e_y) * Y_hat * (1 - Y_hat)
    if(act_func == sigmoid):
      det_func = det_sigmoid
    elif(act_func == relu):
      det_func = det_relu
    elif(act_func == tanh):
      det_func = det_tanh
    elif(act_func == identity):
       det_func = det_identity
    for idx in range(no_hl + 1, 0, -1):
        dA_curr = dA_prev
        grad_val['W' + str(idx)] = np.dot(dA_curr.T, act_val['H' + str(idx - 1)])
        grad_val['B' + str(idx)] = np.sum(dA_curr, axis=0)
        if idx != 1:
            dH_prev = np.dot(dA_curr, params['W' + str(idx)])
            dA_prev = np.multiply(dH_prev, det_func(act_val['A' + str(idx - 1)]))

    return grad_val