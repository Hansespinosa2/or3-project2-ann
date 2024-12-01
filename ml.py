import numpy as np

class ActivationFunctions():
    def __init__(self):
        pass

    @staticmethod    
    def relu(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def relu_derivative(x):
        return  (x > 0).astype(np.float32)
    
    @staticmethod
    def relu_cubed(x):
        return np.power(np.maximum(x,0),3)
    
    @staticmethod
    def relu_cubed_derivative(x):
        return 3 * np.power(np.maximum(x,0),2) * (x > 0).astype(np.float32)
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s (1 - s)
    

class Layer():
    def __init__(self, input_size, output_size, activation = 'relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(self.output_size, self.input_size)
        self.b = np.random.randn(self.output_size, 1)

        if activation == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif activation == 'relu_cubed':
            self.activation = ActivationFunctions.relu_cubed
            self.activation_derivative = ActivationFunctions.relu_cubed_derivative
        elif activation == 'linear':
            self.activation = ActivationFunctions.linear
            self.activation_derivative = ActivationFunctions.linear_derivative
        elif activation == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative

    def forward(self, X):
        self.Z = self.W @ X + self.b
        self.A = self.activation(self.Z)
        self.X = X
        return self.A
    
    def backward(self, dA, gamma):
        dZ = dA * self.activation_derivative(self.Z)

        dW = dZ @ self.X.T / self.X.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / self.X.shape[1]
        dX = self.W.T @ dZ

        self.W -= gamma * dW
        self.b -= gamma * db

        return dX


