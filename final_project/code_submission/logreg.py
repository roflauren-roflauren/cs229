"""
    ATTRIBUTION STATEMENT: CODE PROVIDED AS PART OF CS 229 PROBLEM SET 1. 
"""

import numpy as np

class LogisticRegression:
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n, dtype=np.float32)

        for i in range(self.max_iter):
            grad = self._gradient(x, y)
            hess = self._hessian(x)

            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hess).dot(grad)

            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(i, loss))

            if np.sum(np.abs(prev_theta - self.theta)) < self.eps:
                break

        if self.verbose:
            print('Final theta (logreg): {}'.format(self.theta))
        
    def predict(self, x):
        y_hat = self._sigmoid(x.dot(self.theta))
        return y_hat

    def _gradient(self, x, y):
        m, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        grad = 1 / m * x.T.dot(probs - y)

        return grad

    def _hessian(self, x):
        m, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))
        diag = np.diag(probs * (1. - probs))
        hess = 1 / m * x.T.dot(diag).dot(x)

        return hess

    def _loss(self, x, y):
        hx = self._sigmoid(x.dot(self.theta))
        loss = -np.mean(y * np.log(hx + self.eps) + (1 - y) * np.log(1 - hx + self.eps))

        return loss

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
