## ABOUT THE PROJECT
The goal of this assignment is to implement and use gradient descent (and its variants) with backpropagation for a classification task of MNIST or Fashion-MNIST dataset from scratch using only numpy
Different optimizers supported by this project are :
* sgd
* momentum
* nag
* adam
* nadam
* rmsprop

## EXPLAINATION OF ALL THE PY FILES

### initialization.py 
This file contains functions used to initialize weights and biases used for training using either random or Xavier method.
### cost_acc.py
This file contains functions used to get the cost and accuracy of our trained models 
### activation_function.py
This file contains all the activation functions and thier derivatives supported by this project which are 
* sigmoid
* tanh
* ReLU
* identity
### nn_func.py
This file has the feedforward and backwardpropagation functions which are used to calculate the gradients and y_predictions.
### train_func.py and update.py
These 2 files have the training and update functions corresponding the specific optimizers {all 6 optimizers}.
training function returns the final params(weights and biases), validation_cost_history and validation_accuracy_history over all epochs  
### train.py
We use this function with the set of hyperparameters to train and run our neural network model.

## INSTRUCTIONS ON HOW TO RUN
Create a wandb Account before running train.py file.
Give the api key to your account when prompted.
The following table contains the arguments supported by the train.py file. 
|Name|	Default Value|	Description|
|:----:| :---: |:---:|
|-wp, --wandb_project	|myprojectname	|Project name used to track experiments in Weights & Biases dashboard|
|-we, --wandb_entity|	myname	|Wandb Entity used to track experiments in the Weights & Biases dashboard.|
|-d, --dataset|	fashion_mnist	|choices: ["mnist", "fashion_mnist"]|
|-e, --epochs	|10	|Number of epochs to train neural network.|
|-b, --batch_size|	50	|Batch size used to train neural network.|
|-l, --loss	|cross_entropy	|choices: ["mean_squared_error", "cross_entropy"]|
|-o, --optimizer|	nadam	|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate	|0.0001	|Learning rate used to optimize model parameters|
|-m, --momentum|	0.9	|Momentum used by momentum and nag optimizers.|
|-beta, --beta|	0.9|	Beta used by rmsprop optimizer|
|-beta1, --beta1|	0.9|	Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2|	0.99	|Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon|	0.000001	|Epsilon used by optimizers.|
|-w_d, --weight_decay|	0	|Weight decay used by optimizers.|
|-w_i, --weight_init	|Xavier	|choices: ["random", "Xavier"]|
|-nhl, --num_layers|	5	|Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size|	128	|Number of hidden neurons in a feedforward layer.|
|-a, --activation	|ReLU	|choices: ["identity", "sigmoid", "tanh", "ReLU"]|
