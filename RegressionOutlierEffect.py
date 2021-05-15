"""

a part of this experiment we will train  linear regression on the data (X, Y) with different regularizations
alpha=[0.0001, 1, 100] and observe how prediction hyper plan moves with respect to the outliers

outliers list [(0,2),(21, 13), (-23, -15), (22,14), (23, 14)] in each of tuple the first elemet
is the input feature(X) and the second element is the output(Y)

"""

import matplotlib.pyplot as plt
import warnings

import numpy as np
import scipy as sp
import scipy.optimize

from sklearn.linear_model import SGDRegressor

warnings.filterwarnings("ignore")


def angles_in_ellipse(num, a, b):
    assert (num > 0)
    assert (a < b)
    angles = 2 * np.pi * np.arange(num) / num
    # print("np.pi", np.pi)
    # print(" __",2*np.pi*np.arange(num)/num)
    if a != b:
        e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
        print("e", e)
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size
        res = sp.optimize.root(
            lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
    return angles


class CustomSGDRegressor(object):
    """
    Custom Vanila implementation of SGD classifier with minimum detail

    """

    def __init__(self, loss='squared_loss', penalty='l2', alpha=0.0001, learning_rate=0.01, eta0=0.01, tol=10e-4):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.tol = tol
        self.learning_rate = learning_rate
        self.eta0 = eta0
        ranNum = np.random.rand(2, )
        self.b = ranNum[0]
        self.w = ranNum[1]
        self.lossVal = float("inf")

    def lossFunc(self, Y, y_pred):
        """
        get the Loss value

        """

        if self.loss == 'squared_loss' and self.penalty == 'l2':
            self.lossVal = np.mean((Y - y_pred) ** 2) + self.alpha * np.mean(self.w ** 2)

        return self.lossVal

    def fit(self, X, Y):

        """
          Fit the line or hyperplane on X
        """

        maxIter = 100
        iter = 0
        prev_w = self.w
        prev_b = self.b
        while iter < maxIter:
            y_pred = (self.w * X + self.b)
            prevLoss = self.lossVal
            self.updateStep(Y, y_pred)
            newLoss = self.lossFunc(Y, y_pred)
            if newLoss > prevLoss - self.tol:
                break
            iter += 1

        return self.w, self.b, self.lossVal

    def updateStep(self, Y, y_pred):
        """
          Gradient Descent Update step
          L2 Penalty used for regularization
        """
        grad_b = -2 * np.mean(Y - y_pred)
        self.b = self.b - self.eta0 * grad_b
        # regualarized with l2 penalty
        grad_w = -2 * np.mean(X * (Y - y_pred)) + 2 * self.alpha * np.mean(self.w)
        # update
        self.w = self.w - self.eta0 * grad_w

    def score(self, X, Y):
        """
          Return the coefficient of determination  of the prediction (1 - SSR/SST)

        """
        y_pred = self.w * X + self.b
        SSR = ((Y - y_pred) ** 2).sum()

        SST = ((Y - Y.mean()) ** 2).sum()

        return 1 - (SSR / SST)
        pass

    def predict(self, X):
        """
        Predict the value of X

        """
        return self.w * X + self.b


a = 2
b = 9
n = 50

phi = angles_in_ellipse(n, a, b)

alphas = [0.0001, 1, 100]
outliers = [(0, 2), (21, 13), (-23, -15), (22, 14), (23, 14)]

for idx, alpha in enumerate(alphas):
    plt.figure(figsize=(20, 20))
    X = b * np.sin(phi)
    Y = a * np.cos(phi)
    for id, outlier in enumerate(outliers):
        plt.subplot(3, len(outliers), id + 1)
        # add outlier one by one

        X = np.append(X, outlier[0]).reshape(-1, 1)
        Y = np.append(Y, outlier[1]).reshape(-1, 1)
        # run custom SGD regression
        customReg = CustomSGDRegressor(alpha=alpha)
        customW, customB, customLoss = customReg.fit(X, Y)
        # run sklern SGD
        sklearnSGD = SGDRegressor(alpha=alpha, learning_rate='constant')
        sklearnSGD.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
        sklearnSGDW = sklearnSGD.coef_[0]
        sklearnSGDB = sklearnSGD.intercept_[0]

        customSGDLine = customW * X + customB
        sklearnSGDLine = sklearnSGDW * X + sklearnSGDB
        plt.scatter(X, Y, color='blue')
        plt.plot(X, customSGDLine, color='orange')
        plt.title(f"alpha {alpha},  number of outliers {id}")

    plt.show()
