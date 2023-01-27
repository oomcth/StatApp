
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from numpy.random import exponential
import matplotlib.pyplot as plt
from math import isnan
from math import exp, log
from scipy.optimize import LinearConstraint


class Portfollio():

    def __init__(self, data, n) -> None:
        self.data = data
        self.n = n
        self.dailyReturns = (-self.data.pct_change().
                             dropna(axis=0,
                             how='any',
                             inplace=False))
        self.cov = self.dailyReturns.cov()
        temp = np.mean(self.dailyReturns)
        self.max = max(temp) * 253
        self.min = min(temp) * 253
        print("range = " + str(self.min) + " ; " + str(self.max))

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
        elif(method == "equalRisk"):  # max diversification
            a = np.ones(self.n)
            temp = 0
            print(self.data.tail(1))
            for i in range(self.n):
                a[i] = self.data.iloc[-1][i]
                temp += self.data.iloc[-1][i]
            a /= temp
            self.weights = a
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
        temp = pd.DataFrame.to_numpy(self.data)
        for i in range(self.n):
            if not isnan(temp[(self.data.index[-1] - 1 - date)[0]][i] / temp[-1][i]):
                a.append(temp[(self.data.index[-1] - 1 - date)[0]][i] / temp[-1][i])
            else:
                a.append(0)
        return np.dot(weight, a)

    def plot(self, date1, date2, weight=[]):
        if weight == []:
            weight = self.weights
        a = np.ones((date2 - date1)[0])
        for i in range((date2 - date1)[0]):
            a[i] = (self.evaluate(weight, date1 + i))
        plt.plot(range((date2-date1)[0]), a)

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
        for i, n in enumerate(self.data):
            if n == key:
                return i
        raise "error"

    def W(self, i):
        a = np.zeros(self.n)
        a[i] = self.weights[i]
        return a

    def Variancecontrib(self, key):
        return (np.transpose(self.W(self.KeyN(key))) @ (self.cov @
                self.W(self.KeyN(key))) /
                (np.transpose(self.weights) @ (self.cov) @ self.weights))

    def RiskEntropy(self):
        H = 0
        for i in self.data:
            H += -self.Variancecontrib(i) * log(self.Variancecontrib(i))
        return -exp(H)

    def WeightEntropy(self):
        H = 0
        for i in range(self.n):
            H += -self.weights[i] * log(self.weights[i])
        return -exp(H)

    def minTEPtf(self, A, b1, b2, lb, ub,
                 wBench,
                 lVol=0,
                 uTE=1,
                 is_risk_adapt=False,
                 adapt_vol=1,
                 adapt_te=1000):
        n = self.cov.shape[0]
        init_guess = np.repeat(1/n, n)
        bounds = np.concatenate((lb, ub), axis=1)
        bounds = tuple(map(tuple, bounds))
        linear_ineq = LinearConstraint(A, b1, b2)
        if is_risk_adapt:
            bench_vol = self.portfolio_sd(self.weights)
            lVol_ = (1 - adapt_vol) * bench_vol
            uTE_ = adapt_te * bench_vol
        else:
            lVol_ = lVol
            uTE_ = uTE

        vol_min = {
            'type': 'ineq',
            'args': (self.cov,),
            'fun': lambda weights, matCov:
                self.portfolio_sd(self.weights)-lVol_
        }
        te_max = {
            'type': 'ineq',
            'args': (self.cov, wBench),
            'fun': lambda weights, matCov, wBench:
                    -(self.portfolio_sd(self.weights - wBench)-uTE_)
        }

        constraints = (
            linear_ineq,
            vol_min,
            te_max
        )
        weights = minimize(self.portfolio_sd(self.weights - wBench),
                           init_guess,
                           args=(self.cov, wBench), method='SLSQP',
                           options={'disp': False},
                           constraints=constraints,
                           bounds=bounds)
        return weights.x
