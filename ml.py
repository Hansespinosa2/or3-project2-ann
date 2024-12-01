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
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x),0)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s (1 - s)
    

class InputLayer():
    def __init__(self,input_size):
        self.input_size = input_size

    def forward(self, X):
        if X.ndim == 1:
            return X.reshape(-1,1)
        else:
            return X.T

class HiddenLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(self.output_size, self.input_size)
        self.b = np.random.randn(self.output_size, 1)

    def forward(self, X):
        return self.W @ X + self.b