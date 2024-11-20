import numpy as np

class ActivationFunctions():
    def __init__(self):
        pass

    def relu(self, x):
        return np.maximum(x, 0)
    
    def relu_cubed(self, x):
        return np.power(np.maximum(x,0),3)
    
    def linear(self, x):
        return x
    

class InputLayer():
    def __init__(self,input_size, output_size):
        self.output_size = output_size
        self.W = np.random.rand(self.output_size, input_size)

    def forward(self, X):
        return self.W.T @ X

class HiddenLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.rand(self.output_size, self.input_size)
        self.b = np.random.rand(self.output_size, 1)

    def forward(self, X):
        return self.W.T @ X + self.b