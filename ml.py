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
        return s * (1 - s)
    
class CostFunctions():
    def __init__(self):
        pass

    @staticmethod
    def mse_cost(y_hat, y_true):
        return np.mean(np.square(y_hat - y_true)) / 2
    
    @staticmethod
    def mse_derivative(y_hat, y_true):
        return (y_hat - y_true) / y_hat.shape[1]
    
    @staticmethod
    def l1_reg_mse(y_hat, y_true, reg_lambda, layers):
        mse = np.mean(np.square(y_hat - y_true)) / 2
        l1_reg = reg_lambda * np.sum([np.sum(np.abs(layer.W)) for layer in layers])
        return mse + l1_reg
    
    @staticmethod
    def l1_reg_mse_derivative(y_hat, y_true):
        return (y_hat - y_true) / y_hat.shape[1]

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
    
    def backward(self, dA, gamma, reg_lambda=0.0):
        dZ = dA * self.activation_derivative(self.Z)

        dW = dZ @ self.X.T / self.X.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / self.X.shape[1]
        dX = self.W.T @ dZ
        
        dW = np.clip(dW, -1e3, 1e3)
        db = np.clip(db, -1e3, 1e3)

        # Include L1 regularization in weight updates
        self.W -= gamma * (dW + reg_lambda * np.sign(self.W))
        self.b -= gamma * db

        return dX

class NeuralNetwork():
    def __init__(self, layer_list, activations):
        self.layer_list = layer_list
        self.activations = activations

        self.layers = []

        for i in range(len(layer_list) - 1):
            a_layer = Layer(layer_list[i], layer_list[i+1], self.activations[i])
            self.layers.append(a_layer)

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def train(self, X, y, gamma=0.01, epochs=1000, cost_fn='mse', reg_lambda=0.0):
        if cost_fn == 'mse':
            self.loss_fn = CostFunctions.mse_cost
            self.loss_fn_derivative = CostFunctions.mse_derivative
        elif cost_fn == 'l1_reg_mse':
            self.loss_fn = lambda y_hat, y: CostFunctions.l1_reg_mse(y_hat, y, reg_lambda, self.layers)
            self.loss_fn_derivative = CostFunctions.l1_reg_mse_derivative

        for epoch in range(epochs):
            y_hat = self.forward(X)

            loss = self.loss_fn(y_hat, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}; Loss {loss}")
            
            dA = self.loss_fn_derivative(y_hat, y)
            for layer in reversed(self.layers):
                dA = layer.backward(dA, gamma, reg_lambda)