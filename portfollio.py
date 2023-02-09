
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from numpy.random import exponential
import matplotlib.pyplot as plt
from math import isnan
from math import exp, log
import loadData
from cppfct import c_Variancecontrib


class Portfollio():

    def __init__(self, begin, b, end, e, stocks, crypto, step) -> None:
        self.step = step
        self.loader = loadData.Loader(stocks, crypto, begin, end, b, e, step)
        self.data = self.loader.PriceDate(begin, end)
        self.n = len(self.loader.price.columns) - 1
        self.stocks = stocks
        self.crypto = crypto
        self.dailyReturns = (-self.data[self.stocks + self.crypto].pct_change().
                             dropna(axis=0,
                             how='any',
                             inplace=False))
        self.cov = self.dailyReturns.cov()
        temp = np.mean(self.dailyReturns)
        self.max = max(temp) * 253
        self.min = min(temp) * 253
        print("range = " + str(self.min) + " ; " + str(self.max))
        self.npCov = pd.DataFrame.to_numpy(self.cov)

        self.bases = np.ones(self.n)

    def ChangeWindow(self, begin, end):
        self.data = self.loader.PriceDate(begin, end)
        self.n = len(self.loader.price.columns) - 1
        self.dailyReturns = (-self.data[self.stocks + self.crypto].pct_change().
                             dropna(axis=0,
                             how='any',
                             inplace=False))
        self.cov = self.dailyReturns.cov()
        temp = np.mean(self.dailyReturns)
        self.max = max(temp) * 253
        self.min = min(temp) * 253
        self.npCov = pd.DataFrame.to_numpy(self.cov)

    def optimize(self, method, target=0):
        if(method == "minVarT"):
            if(target > self.max):
                target = self.max
                print("target trop grand")
            elif target < self.min:
                target = self.min
                print("target trop petit")
            constraints = ({'type': 'eq',
                            'fun': lambda x:
                            self.portfolio_returns(x) - target},
                           {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            min_sd_results = minimize(fun=self.portfolio_sd,
                                      x0=self.equalWeights(),
                                      method='SLSQP',
                                      bounds=bounds,
                                      constraints=constraints,
                                      tol=10**-10)
            self.weights = min_sd_results.x
        elif(method == "sharpRatio"):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            max_sharpe_results = minimize(fun=self.sharpe_fun,
                                          x0=self.equalWeights(),
                                          method='SLSQP',
                                          bounds=bounds,
                                          constraints=constraints)
            self.weights = max_sharpe_results.x
        elif(method == "equalRisk"):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            min_sd_results = minimize(fun=self.maxContrib,
                                      x0=self.equalWeights(),
                                      method='SLSQP',
                                      bounds=bounds,
                                      constraints=constraints,
                                      tol=10**-10)
            self.weights = min_sd_results.x
        elif(method == "equalWeights"):
            self.weights = np.ones(self.n) * (1/self.n)
        elif(method == "minVar"):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            min_sd_results = minimize(fun=self.portfolio_sd,
                                      x0=self.equalWeights(),
                                      method='SLSQP',
                                      bounds=bounds,
                                      constraints=constraints,
                                      tol=10**-10)
            self.weights = min_sd_results.x
        elif(method == "equalRisk111"):
            a = np.zeros(self.n)
            for n, i in enumerate(self.stocks + self.crypto):
                if n == 0:
                    a[0] = 1
                else:
                    a[n] = self.npCov[0][0] / self.npCov[n][n]
            s = sum(a)
            self.weights = [i / s for i in a]
            print(sum(self.weights))
        else:
            raise "Wrong objective"

    def portfolio_returns(self, weights):
        return (np.dot(self.dailyReturns.mean(), weights)) * 253

    def portfolio_sd(self, weights):
        return np.sqrt(np.transpose(weights) @
                       (self.cov) @ weights)

    def randomWeights(self):
        temp = exponential(1, self.n)
        return temp / np.sum(temp)

    def equalWeights(self):
        return np.array([1/self.n] * self.n)

    def sharpe_fun(self, weights=[]):
        if(weights == []):
            return (-self.portfolio_returns(self.weights) /
                    self.portfolio_sd(self.weights))
        return - (self.portfolio_returns(weights) / self.portfolio_sd(weights))

    def evaluate(self, weight, date):
        a = []
        temp = pd.DataFrame.to_numpy(self.data[self.crypto + self.stocks])
        for i in range(self.n):
            if not isnan(temp[(self.data.index[-1]  - date)[0]][i] /  temp[0][i]):
                a.append(temp[(self.data.index[-1]  - date)[0]][i] / temp[0][i])
            else:
                a.append(0)
        return np.dot(weight, a)

    def plot(self, date1, date2, weights=[]):
        plt.plot(range((date2-date1)[0]), self.evalVect(date1, date2, weights))

    def evalVect(self, date1, date2, weights=[]):
        if weights == []:
            weights = self.weights
        a = np.ones((date2 - date1)[0])
        for i in range((date2 - date1)[0]):
            a[i] = (self.evaluate(weights, date1 + i))
        return np.flip(a)

    def effiscientFrontier(self):
        target = np.linspace(start=self.min,
                             stop=self.max,
                             num=30)
        Frontier = []
        for target in target:
            self.optimize("minVarT", target)
            Frontier.append((self.weights,
                             self.portfolio_returns(self.weights),
                             self.portfolio_sd(self.weights)))
        Frontier = np.array(Frontier)
        return Frontier

    def KeyN(self, key):
        for i, n in enumerate(self.data[self.stocks + self.crypto]):
            if n == key:
                return i
        print(key)
        raise "error"

    def W(self, i, weights=[]):
        if weights == []:
            weights = self.weights
        a = np.zeros(len(weights))
        a[i] = weights[i]
        return a

    def Variancecontrib(self, key, weights=[]):
        if weights == []:
            weights = self.weights
        return c_Variancecontrib(self.W(self.KeyN(key), weights), weights, self.npCov)

    def RiskEntropy(self):
        H = 0
        for i in self.data[self.crypto + self.stocks]:
            H += -self.Variancecontrib(i) * log(self.Variancecontrib(i))
        return -exp(H)

    def WeightEntropy(self):
        H = 0
        for i in range(self.n):
            H += -self.weights[i] * log(self.weights[i])
        return -exp(H)

    def maxContrib(self, weights):
        return max([self.Variancecontrib(i, weights) for i in self.data[self.stocks + self.crypto]])
