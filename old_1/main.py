import MNIST_loader
import nn

def main():
    training_data, validation_data, test_data = MNIST_loader.load_data_nn()
    
    neural_network = nn.Network([784, 100, 10])
    neural_network.stochastic_gradient_descent(training_data, 30, 10, 3.0, 
                                                test_data)

if __name__ == "__main__":
    main()