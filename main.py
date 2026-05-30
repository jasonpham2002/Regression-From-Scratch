import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.w = None
        self.cost_history = []

    def compute_cost(self, X, y):
        m = X.shape[0]
        y_hat = X @ self.w
        j = 1 / (2 * m) * np.sum((y_hat - y) ** 2)
        return j

    def compute_gradient(self, X, y ):
        m = X.shape[0]
        y_hat = X @ self.w

        error = y_hat - y

        dj_dw = (1/m) * (X.T @ error)
        return dj_dw

    def gradient_descent(self, compute_gradient, X, y, alpha, iters, compute_cost):

        for i in range(iters):
            self.w = self.w - alpha * compute_gradient(X,y)
            cost =  compute_cost(X,y)
            self.cost_history.append(cost)

    def fit(self, X,y):
        #add intercept column to X, compute the size of X through m and n
        m = X.shape[0]
        ones = np.ones((m,1))
        X = np.c_[ones, X]
        n = X.shape[1]

        #initializae w
        self.w = np.zeros((n,1))

        self.gradient_descent(self.compute_gradient, X, y, self.lr, self.n_iters, self.compute_cost)

    def predict(self, X):
        return X @ self.w