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
    
class HiddenLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        W = np.random.rand(self.input_size, self.output_size)
        b = np.random.rand(self.input_size, 1)