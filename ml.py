import numpy as np

class ActivationFunctions():
    def __init__(self):
        pass

    @staticmethod    
    def relu(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def relu_cubed(x):
        return np.power(np.maximum(x,0),3)
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x),0)
    

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