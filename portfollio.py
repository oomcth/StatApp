
# classe gérant un portefeuille


import pandas as pd
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from math import exp, log
import loadData
from cppfct import c_Variancecontrib, dateIndex, datestringdiff


class Portfollio():

    def __init__(self, begin, b, end, e, stocks, crypto, step) -> None:
        self.begin = begin
        self.end = end
        self.step = step  # fréquence des données. exemple 1wk = weekly

        # classe loader qui charge les données
        self.loader = loadData.Loader(stocks, crypto, begin, end, b, e, step)
        # données utilisées par la classe selon la fenetre de temps choisie
        self.data = self.loader.PriceDate(begin, end)

        # vérifie l'intégrité des données
        if self.data.isnull().values.any():
            print("attention présence de NaN")
        print(self.loader.price)
        self.n = len(self.loader.price.columns) - 1  # nombre d'actifs
        self.all = self.loader.all   # liste des actifs

        # calcul le tableau des retours des actifs
        self.dailyReturns = (-self.data[self.all].pct_change().
                             dropna(axis=0,
                             how='any',
                             inplace=False))
        self.cov = self.dailyReturns.cov()  # matrice de covariance
        temp = np.mean(self.dailyReturns)  # calcul les rendements moyen par actifs
        self.max = max(temp)  # rendement maximal
        self.min = min(temp)  # rendement minimal

        # affiche les rendements extrémal
        print("range = " + str(self.min) + " ; " + str(self.max))

        # version numpy de la matrice de covariance
        self.npCov = pd.DataFrame.to_numpy(self.cov)

    # Change la fenetre de temps sur laquelle se base le portefeuille
    # fonctions analogues à innit
    def ChangeWindow(self, begin, end):
        self.begin = begin
        self.end = end
        self.data = self.loader.PriceDate(begin, end)
        self.n = len(self.loader.price.columns) - 1
        self.dailyReturns = (-self.data[self.all].pct_change().
                             dropna(axis=0,
                             how='any',
                             inplace=False))
        self.cov = self.dailyReturns.cov()
        temp = np.mean(self.dailyReturns)
        self.max = max(temp) * 253
        self.min = min(temp) * 253
        self.npCov = pd.DataFrame.to_numpy(self.cov)

    # optimise le portefeuille selon la tratégie 'method'
    def optimize(self, method, target=0):

        # minimise la variance avec un rendement target
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

        # maximise le ratio de Sharp
        elif(method == "sharpRatio"):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            max_sharpe_results = minimize(fun=self.sharpe_fun,
                                          x0=self.equalWeights(),
                                          method='SLSQP',
                                          bounds=bounds,
                                          constraints=constraints)
            self.weights = max_sharpe_results.x

        # méthode equalRisk
        elif(method == "equalRisk"):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            min_sd_results = minimize(fun=self.maxContrib,
                                      x0=self.equalWeights(),
                                      method='SLSQP',
                                      bounds=bounds,
                                      constraints=constraints,
                                      tol=10**-10).x
            self.weights = min_sd_results

        # même poids par actifs
        elif(method == "equalWeights"):
            self.weights = np.ones(self.n) * (1/self.n)

        # minimise la variance du portefeuille
        elif(method == "minVar"):
            print(self.n)
            print(self.cov.shape)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            self.weights = minimize(fun=self.portfolio_sd,
                                    x0=self.equalWeights(),
                                    method='SLSQP',
                                    bounds=bounds,
                                    constraints=constraints,
                                    tol=10**-10).x
        # méthode maxDiv
        elif(method == "maxDiv"):
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for i in range(self.n))
            self.weights = minimize(fun=self.neg_DR,
                                    x0=self.equalWeights(),
                                    method='SLSQP',
                                    bounds=bounds,
                                    constraints=constraints,
                                    tol=10**-10).x
        else:
            raise "Wrong objective"

    # rendements du portefeuille
    def portfolio_returns(self, weights=[]):
        if weights == []:
            weights = self.weights
        a = []
        temp = pd.DataFrame.to_numpy(self.data[self.all])
        for i in range(self.n):
            if not isnan(temp[0][i]):
                a.append(temp[-1][i] / temp[0][i] - 1)
            else:
                return 0
        return (np.dot(weights, a) ** (1/len(self.data.index))) - 1

    # retourne la variance du portefeuille
    def portfolio_sd(self, weights):
        return np.sqrt(np.transpose(weights) @
                       (self.cov) @ weights)

    # retourne la composition de portefeuille ou chaque poid est égal
    def equalWeights(self):
        return np.array([1/self.n] * self.n)

    # fonction de sharp
    def sharpe_fun(self, weights=[]):
        if(weights == []):
            return (-self.portfolio_returns(self.weights) /
                    self.portfolio_sd(self.weights))
        return - (self.portfolio_returns(weights) / self.portfolio_sd(weights))

    # retourne l'évaluation normalisée d'un portefeuille à une date précise
    def evaluate(self, weight, date):
        a = []
        temp = pd.DataFrame.to_numpy(self.data[self.all])
        for i in range(self.n):
            if not isnan(temp[(self.data.index[-1] - 1 - date)[0]][i] / temp[0][i]):
                a.append(temp[(self.data.index[-1] - 1 - date)[0]][i] / temp[0][i])
            else:
                return 0
        return np.dot(weight, a)

    # plot la valeur du portefeuille au cours du temps
    def plot(self, date1, date2, weights=[]):
        temp = self.evalVect(date1, date2, weights)
        plt.plot(range(len(temp)), temp)

    # calcul la valeur du portefeuille au cours du temps
    def evalVect(self, date1, date2, weights=[]):
        if weights == []:
            weights = self.weights

        a = np.ones((date2 - date1)[0])
        for i in range((date2 - date1)[0]):
            temp = self.evaluate(weights, date1 + i)
            if not(isnan(temp)):
                a[i] = self.evaluate(weights, date1 + i)
            else:
                a[i] = 0
        return np.flip(a)

    # calcul la frontière efficasse du portefeuille
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
        Frontier = np.array(Frontier, dtype='object')
        return Frontier

    # retourne le numéro de colonne correspondant à un actif
    def KeyN(self, key):
        for i, n in enumerate(self.data[self.all]):
            if n == key:
                return i
        print(key)
        raise "error"

    # retourne un vecteur ou toute les valeur sont nulle sauf la valeur i qui
    # correspond au poid de l'actif i
    def W(self, i, weights=[]):
        if weights == []:
            weights = self.weights
        a = np.zeros(len(weights))
        a[i] = weights[i]
        return a

    # calcul de la contribution à la variance
    def Variancecontrib(self, key, weights=[]):
        if weights == []:
            weights = self.weights
        return c_Variancecontrib(self.W(self.KeyN(key),
                                 weights),
                                 weights,
                                 self.npCov)

    # calcul l'entropie de risque
    def RiskEntropy(self):
        H = 0
        for i in self.data[self.all]:
            H += -self.Variancecontrib(i) * log(self.Variancecontrib(i))
        return -exp(H)

    # calcul l'entropie de poid
    def WeightEntropy(self):
        H = 0
        for i in range(self.n):
            H += -self.weights[i] * log(self.weights[i])
        return -exp(H)

    # retourne le maximum des contribution à la variance des actifs du
    # portefeuille
    def maxContrib(self, weights):
        return max([self.Variancecontrib(i, weights) for i in self.data[self.all]])

    # calcule le diversification ratio
    def DR(self, weights=[]):
        if weights == []:
            weights = self.weights
        weighted_vols = np.sqrt(np.diag(self.npCov)) @ weights
        ptf_vol = np.sqrt(weights.T @ self.npCov @ weights)
        return weighted_vols/ptf_vol

    # calcul l'opposé du diversification ratio
    def neg_DR(self, weights):
        return -self.DR(weights)

    def info(self):
        print("assets :", self.loader.price.columns)
        print("weights :", self.weights)
        print("return : ", self.portfolio_returns(self.weights))
        print("sd : ", self.portfolio_sd(self.weights))
