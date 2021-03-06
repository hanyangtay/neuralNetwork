import numpy as np
import random
import json

#activation function

class Sigmoid:
    @staticmethod
    def fn(z):
        return 1.0/(1.0 + np.exp(-z))
    
    @staticmethod
    def deriv(z):
        return Sigmoid.fn(z) * (1 - Sigmoid.fn(z))

class CrossEntropyCost:
    """
    Cost function
    measures prediction error
    
    y=0,
        a=0, cost = log(1-a) = log 1 = 0
        a=1, cost = log 0 = infinity
    y=1, 
        a=0, cost = -log(a) = infinity
        a=1, cost = log 1 = 0
    """
    
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def deriv(z, a, y):
        return (a-y)
    
class QuadraticCost:
    """
    Cost function
    measures prediction error
    
    cost = 1/2 * sum (a-y)**2
    """
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def deriv(z, a, y):
        ########## ask for help
        return (a-y) * Sigmoid.deriv(z)
        
class Network:
    
    def __init__(self, sizes, activate=Sigmoid, cost=QuadraticCost):
        """
        Sizes refer to the number of neurons in each layer.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activate = activate
        self.cost = cost

        """
        Random bias and weight initialisation
        No bias is needed for input parameters.
        
        Weights are divided by number of input neurons.
        Ensures that weighted sum (z) has a standard deviation of ~1.
        Note that z is a sum of x input features/neurons.
        
        Biases is a list of column vectors, corresponding to each output neuron.
        
        Velocities (used for momentum) and weights are lists of matrices.
        Row of each matrix: output neurons
        Column of each matrix: Input neurons to each output neuron
        """
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.velocities = [np.zeros((y,x)) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        return
      
    def stochastic_gradient_descent(self, training_data, test_data, epochs,
        mini_batch_size, learning_rate, reg_lambda = 0.0, momentum = 1.0,
                                   eval_train = True, eval_test = True):
        """
        Learning algorithm to minimise cost function
        
        eval_train reveals performance of nn over training data
        Primarily for debugging purposes
        Set it to false to train the nn faster without showing performance
        """
        
        n = training_data[1].shape[1]
        n_tests = test_data[1].shape[1]
        
        #initialise values
        training_cost = []
        training_accuracy = []
        test_cost = []
        test_accuracy = []

        for j in range(epochs):
            
            #shuffle training data
            seed = np.random.permutation(n)
            training_data[0] = np.transpose(np.transpose(training_data[0])[seed])
            training_data[1] = np.transpose(np.transpose(training_data[1])[seed])
            
            #split into mini-batches
            mini_batches = [(training_data[0][:, k:k+mini_batch_size],
                            training_data[1][:, k:k+mini_batch_size])
                            for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                #Gradient descent on each mini-batch
                self.update_mini_batch(mini_batch, learning_rate, reg_lambda, 
                                       momentum, mini_batch_size, n)
            
            print ("Epoch {} completed.".format(j+1))
            
            if eval_train:
                cost = self.total_cost(training_data, n, reg_lambda)
                accuracy = self.accuracy(training_data)
                training_cost.append(cost)
                training_accuracy.append(accuracy)
                print ("Cost on training data: {}".format(cost))
                print ("Accuracy on training data: {} / {} \n".format(accuracy, 
                                                                      n))
            
            if eval_test:
                cost = self.total_cost(test_data, n_tests, reg_lambda)
                accuracy = self.accuracy(test_data)
                test_cost.append(cost)
                test_accuracy.append(accuracy)
                print ("Cost on test data: {}".format(cost))
                print ("Accuracy on test data: {} / {}\n".format(accuracy, 
                                                                 n_tests))
            
        return training_cost, training_accuracy, test_cost, test_accuracy
                
    def update_mini_batch(self, mini_batch, learning_rate, reg_lambda,
                          momentum, mini_batch_size, n):
        """
        Apply gradient descent using backpropagation to a mini batch.
        L2 regularisation - prevent overfitting by keep weights small
        Update network's weights and biases.
        
        Each training example is reshaped into a vector column.
        X and Y are matrices of the entire mini batch, comprising parameters
        and outputs of each training example respectively.
        """
        
        mini_batch_X = mini_batch[0]
        mini_batch_Y = mini_batch[1]
        
        delta_b, delta_w = self.backprop(mini_batch_X, mini_batch_Y, 
                                         mini_batch_size)

        self.velocities = [ momentum * v - learning_rate / mini_batch_size * new_v
                           for v, new_v in zip(self.velocities, delta_w)]

        self.weights = [(1 - learning_rate*(reg_lambda/n)) * w + v
                        for w, v in zip(self.weights, self.velocities)]
        
        self.biases = [b - learning_rate / mini_batch_size * new_b
                        for b, new_b in zip(self.biases, delta_b)]
            
    def backprop(self, x, y, mini_batch_size):
        """
        Backpropagation
        Aim: Change weights and biases to reduce cost
        
        delta   = error of neuron
                = dCost wrt weighted input, z
                  (if this is small, cost won't change much, 
                  and is near optimal)
                  
        *******1*******  
        delta of output neuron
        delta   = dCost wrt activation, a * da wrt z (chain rule)
                = dCost wrt a * activation_function_deriv(z)
                
        *******2******* 
        delta of hidden layer neuron
        delta   = dCost wrt z_nextlayer * dz_nextlayer wrt z 
                = delta_nextlayer * dz_nextlayer wrt z
                = w_nextlayer * activation_function_deriv(z) * delta_nextlayer
                
        Note that z_nextlayer = w_nextlayer * a + b_nextlayer,
        differentiating it with chain rule yields
        
        *******3******* 
        delta_b
        dCost wrt bias  = dCost wrt bias * dbias wrt z
                        = dCost wrt z
                        = delta
        
        Note that z = w*x + bias, so bias = z - w*x, dbias wrt z = 1      
        
        *******4*******  
        delta_w
        dCost wrt weights   = dCost wrt z * dz wrt weights
                            = delta * a_in
        
        Note that z = weights * a_in + b
                  
        """
        #initialise values
        delta_b = [0 for b in self.biases]
        delta_w = [0 for w in self.weights]
        
        #forward propagation
        activation = x
        
        #stores each layer of activations and weighted inputs
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activate.fn(z)
            activations.append(activation)
        
        #backprop
        
        delta = self.cost.deriv(zs[-1], activations[-1], y)
        delta_b[-1] = delta.sum(1).reshape([len(delta), 1])
        delta_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.activate.deriv(z)
            delta_b[-l] = delta.sum(1).reshape([len(delta), 1])
            delta_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (delta_b, delta_w)
    
    def forward_prop(self, a):
        for b, w in zip(self.biases, self.weights):
                a = self.activate.fn(np.dot(w,a) + b)
        return a
        
    
    def accuracy(self, data):
        """
        Forward propagation, then compare results.
        
        Returns number of cases that are correctly classified.
        """
        
        predicted = np.argmax(self.forward_prop(data[0]), axis=0), 
        actual = np.argmax(data[1], axis=0)
                   
        
        return np.sum(predicted == actual)
        
    def total_cost(self, data, n, reg_lambda):
        """
        Computes cost of model over a data set.
        """
        
        cost = 0.0
        
        a = self.forward_prop(data[0])
        cost = np.sum(self.cost.fn(a, data[1])) / n
        
        #regularisation
        cost += 0.5 * (reg_lambda / n) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def save(self, filename):
        """
        Save neural network
        """
        
        data = {    "sizes": self.sizes,
                    "weights": [w.tolist() for w in self.weights],
                    "biases": [b.tolist() for b in self.biases],
                    "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
        return
        
    def load(filename):
        """
        Load neural network
        """
        
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        cost = getattr(sys.modules[__name__], data["cost"])
        nn = Network(data["sizes"], cost = cost)
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases = [np.array(b) for b in data["biases"]]
        
        return nn
    


        