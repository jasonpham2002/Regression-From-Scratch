import numpy as np
import pandas as pd

class LinearRegressionFS:
    def __init__(self, learning_rate=0.001, n_iterations=3000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.w = None
        self.cost_history = []

    def compute_cost(self, X, y):
        m = X.shape[0]
        y_hat = X @ self.w
        j = 1 / (2 * m) * np.sum((y_hat - y) ** 2) # compute cost function
        return j

    def compute_gradient(self, X, y ):
        m = X.shape[0]
        y_hat = X @ self.w

        error = y_hat - y

        dj_dw = (1/m) * (X.T @ error) #compute gradient
        return dj_dw

    def gradient_descent(self, compute_gradient, X, y, alpha, iters, compute_cost):

        for i in range(iters):
            self.w = self.w - alpha * compute_gradient(X,y)
            cost =  compute_cost(X,y)
            self.cost_history.append(cost)

    def fit(self, X,y):
        #add intercept column to X, compute the size of X through m and n

       # X = X.to_numpy()
        m = X.shape[0]
        ones = np.ones((m,1))
        X = np.c_[ones, X]
        n = X.shape[1]

        y = y.to_numpy().reshape(-1,1).astype(np.float64)

        #initializae w
        self.w = np.zeros((n,1), dtype = np.float64)

        # convert y from pandas dataframe to a numpy array


        self.gradient_descent(self.compute_gradient, X, y, self.lr, self.n_iters, self.compute_cost)

    def predict(self, X):
      #  X = X.to_numpy()
        m = X.shape[0]
        ones = np.ones((m, 1))
        X = np.c_[ones, X]

        return X @ self.w


    def accuracy(self, y_pred, y_true):
        # calculation R^2 (coefficient of determination)
        y_true = y_true.to_numpy().reshape(-1,1)
        y_mean = np.mean(y_true)
        co_deter = (np.sum((y_true - y_pred)**2)) / (np.sum((y_true - y_mean)**2))

        return 1 - co_deter

class LogisticRegressionFS:
    def __init__(self, learning_rate=0.001, n_iterations = 3000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.w = None
        self.cost_history = []


    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/ (1 + np.exp(-z))

    def compute_cost(self, X, y):
        m = X.shape[0]

        y_hat = self._sigmoid(X @ self.w) # compute prediction from weights, converted prediction to probability
                                        # using sigmoid function
        y_hat = np.clip(y_hat,1e-15, 1 - (1e-15))
        j = -1/m * np.sum(y * np.log(y_hat) + (1-y)*np.log(1 - y_hat)) #compute cost function
        return j

    def compute_gradient(self, X, y):
        m = X.shape[0]
        y_hat = self._sigmoid(X @ self.w)

        dj_dw = 1/m * ( X.T @ (y_hat-y)) # compute gradient
        return dj_dw

    def gradient_descent(self, X, y):
        for i in range(self.n_iters):

            self.w = self.w - self.lr * self.compute_gradient(X,y)
            self.cost_history.append(self.compute_cost(X,y))



    def fit(self, X, y):
        m = X.shape[0] #number of training examples
        ones = np.ones((m,1))

        X = np.c_[ones, X] # added intercept weight
        n = X.shape[1] # number of weights

        self.w = np.zeros((n,1))


        self.gradient_descent(X,y)

    def predict(self, X, threshold = 0.5):

        m = X.shape[0]  # number of testing examples
        ones = np.ones((m, 1))

        X = np.c_[ones, X]  # added intercept weight
        n = X.shape[1]  # number of weights

        ypredict = X @ self.w
        z = self._sigmoid(ypredict)
        return (z>=threshold).astype(int)

    def accuracy_test(self, y_predict, y_true):
        return np.mean(y_predict == y_true)