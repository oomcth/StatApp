
# classe du robot trader

import portfollio
import datetime
from cppfct import dateTimeToDate
import matplotlib.pyplot as plt
from cppfct import dateIndex


class Robot:

    def __init__(self, begin, b, end, e, stocks, crypto, datastep):
        # date de début des données
        tempb = datetime.date(b.year - 2, b.month, b.day)
        # date de début de l'activité du robot
        tempbegin = str(tempb)

        # création du portefeuille
        self.portfollio = portfollio.Portfollio(tempbegin,
                                                tempb,
                                                end,
                                                e,
                                                stocks,
                                                crypto,
                                                datastep)
        self.step = "3month"  # fréquence à laquelle le robot se met à jour

    # défini la stratégie du Robot
    def setStrategie(self, strategie):
        self.strategie = strategie

    # défini la fréquence des données
    def setStep(self, step):
        self.step = step

    # simule l'activité du robot
    def RUN(self, begin, b, end, e):
        days = (e-b).days  # nombre de jour dans la simulation
        nsteps = days / self.dayInStep(self.step)  # nombre de mises à jour
        nsteps = int(nsteps)  # nombre de mise à jour du robot (au format int)
        self.eval = [1]  # évaluation au cours du temps du portefeuille

        # boucle principale
        for i in range(nsteps):
            # calcul les dates d'activité sans mise à jour du robot
            tempb = max(b + datetime.timedelta(days=i*self.dayInStep(self.step)), b)
            tempe = min(b + (datetime.timedelta(days=1+(i+1)*self.dayInStep(self.step))), e)
            tempbegin = dateTimeToDate(tempb)
            tempend = dateTimeToDate(tempe)
            a = (datetime.date(int(str.split(str(tempbegin), "-")[0]),
                               int(str.split(tempbegin, "-")[1]),
                               int(str.split(tempbegin, "-")[2])) - datetime.timedelta(days=52*30))

            # calibre le portefeuille sur la plage de temps étudiée
            self.portfollio.ChangeWindow(str(a), tempbegin)

            # optimise le portefeuille
            self.portfollio.optimize(self.strategie)

            # calibre le portefeuille sur la plage de temps simulée
            self.portfollio.ChangeWindow(tempbegin, tempend)

            # évalue le portefeuille au cours du temps
            self.eval += [i * self.eval[-1] for i in (self.portfollio.evalVect(
                dateIndex(self.portfollio.loader.price, tempbegin),
                dateIndex(self.portfollio.loader.price, tempend)))]

    # affiche la valeur du portefeuille du robot au cours du temps
    def plot(self):
        plt.plot(range(len(self.eval) - 1), self.eval[2:])

    # affiche la valeur du portefeuille du robot au cours du temps
    def show(self):
        plt.show()

    # calcul le nombre de jour corespondant à la fréquence des données
    @staticmethod
    def dayInStep(step):
        if step == "day":
            return 1
        elif step[1:] == "week":
            return 7 * int(step[0])
        elif step[1:] == "month":
            return 30 * int(step[0])
        elif step[1:] == "year":
            return 365 * int(step[0])
