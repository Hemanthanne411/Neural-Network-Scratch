import pandas as pd
import numpy as np
import copy



class NeuralNetwork:
    """
    This class contains all the helper functions to implement forward propagation and 
    backward propagation of a Neural Network.
    """
    def __init__(self, n_h, activations, iterations= 800, learning_rate = 0.01, print_cost=False, seed=1):
        np.random.seed(seed)
        self.n_h = n_h
        self.activations= activations
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost


    def _parameter_initialize(self, L):
        parameters = {}

        for i in range(len(L) - 1):
            parameters["W" + str(i+1)] = np.random.randn(L[i+1], L[i]) * 0.05
            parameters["b" + str(i+1)] = np.zeros((L[i+1], 1))

        return parameters

    def _forward_linear(self, A, W, b):

        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def _activation_functions(self, Z, act_fn):
        if act_fn == 'relu':
            return np.maximum(0, Z), Z
        
        elif act_fn == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
            return A, Z
        elif act_fn == 'softmax':
            Z_shifted = Z - np.max(Z, axis=0, keepdims=True)  # for numerical stability
            exp_Z = np.exp(Z_shifted)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            return A, Z
        # ADD MORE ACTIVATION FUNCTIONS

    def _forward_activation(self, A_prev, W, b, activation):
        """ This Function calculates the forward prop for one layer and stores the activation and cache containin (A_prev, W, b)"""

        # capture both Z and also the A,W,b
        Z, linear_cache = self._forward_linear(A_prev, W, b)
        # activation_cache captures the Z value     
        A, activation_cache = self._activation_functions(Z, activation)
        
        
        cache = (linear_cache, activation_cache)
        # cache = [
        #          ((A, W, b), Z)
        #          ]

        return A, cache

    def _forward_propagation(self, parameters, X, activations):
        """ FWD prop """
        A = X
        L = len(parameters) // 2
        caches = []
        for l in range(1, L):
            A_prev = A
            A, cache = self._forward_activation(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = activations[0])
            caches.append(cache)
            

        AL, cache  = self._forward_activation(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = activations[1])
        caches.append(cache)

        #  caches  = [
        #             ((A, W, b), Z) ---> Layer 1
        #             ((A, W, b), Z) ---> layer 2
        #            .
        #            .
        #            .
        #            ((A, W, b), Z) ---> layer L
        #            ]
        
        return AL, caches

    def _compute_cost(self, AL, Y):
        """
        Computes Cost using Categorical Cross Entropy Loss Function
        Cost = (-1/m) * Sum for all examples -[Y*log(A)]
        
        """
        m = Y.shape[1]

        logar = np.multiply(Y, np.log(AL))
        
        cost = (np.sum(logar))/(-m)
        print(f"AL (example 3): {AL[:, 2]}")
        print(f"Y (example 3): {Y[:, 2]}")
        print(f"Sum of softmax probabilities for example 3: {np.sum(AL[:, 2])}")
        # cost = np.squeeze(cost) # to return just correct cost shape (float)

        # if iterations % 100 ==0 and print_cost == True:
        #     print(f"Iteration: {iterations}./n Cost : {cost}") 

        return cost
    
    def _activation_derivatives(self, activation, cache):

        Z = cache
        if activation == 'sigmoid':
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        
        elif activation == 'relu':
            dZ = np.array(Z > 0, dtype=float)
            return dZ
        elif activation == 'tanh':
            tanh = np.tanh(Z)
            dZ = 1 - tanh**2
            return dZ
        elif activation == 'leaky_relu':
            dZ = np.where(Z > 0, 1, 0.01)
            return dZ
        elif activation == 'softmax':
            # Softmax derivative is usually handled together with cross-entropy loss,
            # but if needed, return Jacobian (only for academic or special purposes)
            # This is a simplified softmax derivative per element (diagonal of Jacobian)
            exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            softmax = exps / np.sum(exps, axis=0, keepdims=True)
            dZ = softmax * (1 - softmax)  # Only valid if used element-wise
            return dZ
        
    def _linear_backpropagation(self, dZ, cache):
        """ This Function computes the linear backward gradients which is the same irrespective of activation functions
        as Z = (W).(A_prev) + (b), '() -linear relationship' 
        """

        A_prev, W, b = cache
        # No of inputs from prev activation
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis = 1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db
    
    def _activation_backpropagation(self, dA, cache, activation):

        linear_cache,activation_cache = cache

        # Passing activation cache which is Z
        dZ = np.multiply(dA, self._activation_derivatives(activation, activation_cache))
        # Passing linear cache which is A, W, b
        dA_prev, dW, db = self._linear_backpropagation(dZ, linear_cache)

    
        # More activations can be defined
        
        return dA_prev, dW, db
    
    def _back_propagation(self, AL, Y, caches, activations):
        """ Calculates the gradient values for each parameter and returns a dictionary of them """

        gradients = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        if activations[1] == "softmax":
            dAL = AL - Y  
            dZ = dAL                                           # for softmax + categorical crossentropy
        elif activations[1] == "sigmoid":
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    # for sigmoid + binary crossentropy
        else:
            raise ValueError(f"Unsupported output activation: {activations[1]}")

        # # Derivative for the final output layer activation with CATEGORICAL CROSS ENTROPY LOSS
        # epsilon = 1e-12
        # AL = np.clip(AL, epsilon, 1. -epsilon)
        # dAL = -(np.divide(Y, AL))                        # Some formula depending on the Cost Function.

        # Index 0, 1, 2, 3... L so last layer is L-1                          
        last_cache = caches[L-1]
        # Output layer( L'th layer) gradients take dAL, cache containing (A_prev, W, b), activation of the ouput.
        # This is a single seperate computation because of unique dAL.
        linear_cache,activation_cache = last_cache
        # dA_prev_temp, dW_temp, db_temp = _activation_backpropagation(dAL, last_cache, activation = activations[1])
        dA_prev_temp, dW_temp, db_temp = self._linear_backpropagation(dZ, linear_cache)

        gradients["dA" + str(L-1)] = dA_prev_temp
        gradients["dW" + str(L)] = dW_temp
        gradients["db" + str(L)] = db_temp


        # Now we have the dA_prev from the last layer and this can be used to compute all the gradients from (L-1)layer
        # in a single for loop because of the same activation.
        for l in reversed(range(L-1)):
            cache = caches[l]
            # From L-1 layer we have the same back prop till the first layer.
            # The reversed range gives L-2, L-3, ..... 2,1,0.
            dA_prev_temp, dW_temp, db_temp = self._activation_backpropagation(gradients["dA" + str(l+1)], cache=cache, activation = activations[0])
            gradients["dA" + str(l)] = dA_prev_temp
            gradients["dW" + str(l+1)] = dW_temp
            gradients["db" + str(l+1)] = db_temp

        # gradients = {
        #     "dW2": dW2,
        #     "db2": db2,
        #     "dW1": dW1,
        #     "db1": db1,
        # }
        return gradients
    

    def _gradient_descent(self, parameters, grads, learning_rate = 0.02):
        """ This function applies gradient descent, updates the weight and bias parameters using their respective gradients.
        
        
        Returns - parameters
        """

        parameters = copy.deepcopy(parameters)
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l +1)] = parameters["W" + str(l +1)] - (learning_rate * grads["dW" + str(l+1)])
            parameters["b" + str(l +1)] = parameters["b" + str(l +1)] - (learning_rate * grads["db" + str(l+1)])
        
        # parameters["W1"] -= learning_rate * dW1
        # parameters["b1"] -= learning_rate * db1 

        return parameters
    
    def train_NN(self, X, Y):
        """ This function takes the input X and output label Y and performs
                -forward propagation,
                -computes cost, and applies 
                -back propagation to calculate the gradients and 
                -updates the parameters 
                
        """

        L = [X.shape[0], *self.n_h]
        parameters = self._parameter_initialize(L)
        for iter in range(self.iterations):

            AL, caches = self._forward_propagation(parameters, X, activations = self.activations)
            # Print the cost for every 100 iterations.
            if iter % 100 == 0 and self.print_cost == True:
                cost = self._compute_cost(AL, Y)
                print("Cost after iteration {}: {}".format(iter, np.squeeze(cost))) 
            # Computing gradients
            grads = self._back_propagation(AL, Y, caches, self.activations)
            # Parameter update
            parameters = self._gradient_descent(parameters, grads, self.learning_rate)
            

        return parameters
    
    def predict_y(self, X: np.ndarray, parameters: dict[str, np.ndarray], activations) -> np.ndarray:

        AL,caches = self._forward_propagation(parameters, X, activations= activations)
        y_pred = np.argmax(AL, axis = 0)
        return y_pred
    
    def accuracy(self, true_y, predicted_y):
        return np.mean(true_y == predicted_y)
        # print("ACCURACY :", accuracy)
        # print(f"Your Neural Networks Prediction is {predicted_y}")
        # print(f"The actual Label of the given input is {true_y}")