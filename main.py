import MNIST_loader
import nn

def main():
    training_data, validation_data, test_data = MNIST_loader.load_data_nn()
    
    #784 parameters, 100 hidden neurons, 10 output neurons
    neural_network = nn.Network([784, 100, 10])
    
    #30 epochs, mini-batches of 10, 0.5 learning rate, 1.0 lambda, 0.0 momentum
    neural_network.stochastic_gradient_descent(training_data, test_data, 30, 10,
                                               0.5, 1.0, 0.0)

if __name__ == "__main__":
    main()