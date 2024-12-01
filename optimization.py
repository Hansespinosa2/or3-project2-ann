import numpy as np

class CostFunctions():
    def __init__(self):
        pass

    @staticmethod
    def mse_cost(y_hat,y_true):
        y_true = y_true.reshape(1,-1)
        return np.sum(np.square(y_hat-y_true),axis=1) / (2*y_hat.shape[1])
