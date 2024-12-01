import numpy as np

class CostFunctions():
    def __init__(self):
        pass

    @staticmethod
    def mse_cost(y_hat,y_true):
        return np.mean(np.square(y_hat-y_true),axis=1) / 2
    
    @staticmethod
    def mse_derivative(y_hat, y_true):
        return (y_hat-y_true) / y_hat.shape[1]
        
