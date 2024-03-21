import argparse
import numpy as np
import wandb
from tqdm import tqdm

# Import necessary functions from the provided modules
from initialization import init_layers, init_layers_xavier, init_u
from train_func import (
    train_sgd,
    train_momentum,
    train_nestrov,
    train_rmsprop,
    train_adam,
    train_nadam
)
from activation_functions import sigmoid, relu, tanh,identity
from nn_func import feedforward
# Define available optimizers
optimizers = {
    'sgd': train_sgd,
    'momentum': train_momentum,
    'nag': train_nestrov,
    'rmsprop': train_rmsprop,
    'adam': train_adam,
    'nadam': train_nadam
}
activation_func = {
    "identity": identity,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "ReLU": relu
}

def load_dataset(dataset):
    if dataset == "mnist":
        # Load MNIST dataset implementation
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == "fashion_mnist":
        # Load Fashion MNIST dataset implementation
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset name. Please choose 'mnist' or 'fashion_mnist'.")
    return (x_train, y_train), (x_test, y_test)

def train(args):
    # Initialize WandB
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    wandb.login()
    # Load dataset (assuming it's loaded as X, Y, x_test, y_test)
    (X,Y),(x_test,y_test)=load_dataset(args.dataset)
    # Perform training
    optimizer = optimizers[args.optimizer]
    chosen_activation_func = activation_func[args.activation]
    params,cost_history,accuracy_history = optimizer(
        X, Y, x_test, y_test,
        args.num_layers, args.hidden_size, args.epochs,
        args.learning_rate, args.batch_size, args.weight_decay,
        chosen_activation_func, init_layers if args.weight_init == 'random' else init_layers_xavier,
        args.loss,args.beta,args.beta1,args.beta2,args.epsilon,args.momentum
    )
    yh_best,_ =feedforward(x_test.reshape(10000,784),params,args.num_layers,args.hidden_size,chosen_activation_func)
    yh_best = np.argmax(yh_best,axis=1)
    class_names =  ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                            y_true=y_test,
                                                            preds=yh_best,
                                                            class_names=class_names,
                                                            title="Confusion Matrix",
                                                            )})

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Neural Network Training Script")
    parser.add_argument("--wandb_entity", "-we", default="myname", help="Wandb Entity")
    parser.add_argument("--wandb_project", "-wp", default="myprojectname", help="Wandb Project Name")
    parser.add_argument("--dataset", "-d", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset")
    parser.add_argument("--epochs", "-e", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", default=150, type=int, help="Batch size")
    parser.add_argument("--loss", "-l", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("--optimizer", "-o", default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("--learning_rate", "-lr", default=0.005, type=float, help="Learning rate")
    parser.add_argument("--momentum", "-m", default=0.9, type=float, help="Momentum for optimizers using it")
    parser.add_argument("--beta", "-beta", default=0.9, type=float, help="Beta for optimizers using it")
    parser.add_argument("--beta1", "-beta1", default=0.9, type=float, help="Beta1 for optimizers using it")
    parser.add_argument("--beta2", "-beta2", default=0.99, type=float, help="Beta2 for optimizers using it")
    parser.add_argument("--epsilon", "-eps", default=0.000001, type=float, help="Epsilon for optimizers using it")
    parser.add_argument("--weight_decay", "-w_d", default=0.0, type=float, help="Weight decay for optimizers using it")
    parser.add_argument("--weight_init", "-w_i", default="Xavier", choices=["random", "Xavier"], help="Weight initialization")
    parser.add_argument("--num_layers", "-nhl", default=5, type=int, help="Number of hidden layers")
    parser.add_argument("--hidden_size", "-sz", default=128, type=int, help="Hidden layer size")
    parser.add_argument("--activation", "-a", default="ReLU", choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function")

    args = parser.parse_args()

    # Perform training
    train(args)

if __name__ == "__main__":
    main()
