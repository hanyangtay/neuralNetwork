import numpy as np
import random

#activation function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
        
def sigmoid_deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))
        
class Network:
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        #random initialisation of biases and weights
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
                        
    def forward_prop(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a
      
    def stochastic_gradient_descent(self, training_data, epochs,
                                    mini_batch_size, learning_rate, 
                                    test_data):
        
        n_tests = test_data[0].shape[1]
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            
            print ("Epoch {0}: {1} / {2}".format(j + 1, self.evaluate(test_data), 
                                                        n_tests))
                
    def update_mini_batch(self, mini_batch, learning_rate):
        batch_size = len(mini_batch)
        
        mini_batch_X = np.hstack((np.reshape(x, (784, 1)) for x, y in mini_batch))
        mini_batch_Y = np.hstack((y for x, y in mini_batch))
        
        delta_b, delta_w = self.backprop(mini_batch_X, mini_batch_Y)
        
        """
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x,y)
            update_b = [new_b + deriv_b for new_b, deriv_b 
                            in zip(update_b, delta_b)]
            update_w = [new_w + deriv_w for new_w, deriv_w 
                            in zip(update_w, delta_w)]
        """
        self.weights = [w - learning_rate * (new_w / batch_size)
                        for w, new_w in zip(self.weights, delta_w)]
        self.biases = [b - learning_rate * (new_b / batch_size)
                        for b, new_b in zip(self.biases, delta_b)]
            
    def backprop(self, x, y):
        
        #initialise values
        delta_b = [0 for b in self.biases]
        delta_w = [0 for w in self.weights]
        
        #forward propagation
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #backprop    
        delta = self.cost_derivative(activations[-1], y) * sigmoid_deriv(zs[-1])
        delta_b[-1] = delta.sum(1).reshape([len(delta), 1])
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_deriv(z)
            delta_b[-l] = delta.sum(1).reshape([len(delta), 1])
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (delta_b, delta_w)
        
    #returns a vector of partial derivatives for the output activations 
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    #returns prediction results
    def evaluate(self, test_data):
        test_results = (np.argmax(self.forward_prop(test_data[0]), axis=0),
                        test_data[1])
        return sum(test_results[0] == test_results[1])
    