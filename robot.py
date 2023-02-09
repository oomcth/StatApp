import portfollio
import datetime
from cppfct import dateTimeToDate
import matplotlib.pyplot as plt
from cppfct import dateIndex


class Robot:

    def __init__(self, begin, b, end, e, stocks, crypto, datastep):
        tempb = datetime.date(b.year - 2, b.month, b.day)
        tempbegin = str(tempb)

        self.portfollio = portfollio.Portfollio(tempbegin, tempb, end, e, stocks, crypto, datastep)
        self.step = "3month"

    def setStrategie(self, strategie):
        self.strategie = strategie

    def setStep(self, step):
        self.step = step

    def RUN(self, begin, b, end, e):
        days = (e-b).days
        nsteps = days / self.dayInStep(self.step)
        nsteps = int(nsteps)
        self.eval = [1]
        for i in range(nsteps):
            tempb = max(b + datetime.timedelta(days=i*self.dayInStep(self.step)), b)
            tempe = min(b + (datetime.timedelta(days=1+(i+1)*self.dayInStep(self.step))), e)
            tempbegin = dateTimeToDate(tempb)
            tempend = dateTimeToDate(tempe)
            a = (datetime.date(int(str.split(str(tempbegin), "-")[0]),
                int(str.split(tempbegin, "-")[1]),
                int(str.split(tempbegin, "-")[2])) - datetime.timedelta(days=52*30))
            self.portfollio.ChangeWindow(str(a), tempbegin)
            self.portfollio.optimize(self.strategie)
            print(self.portfollio.weights)
            self.portfollio.ChangeWindow(tempbegin, tempend)
            self.eval += [i * self.eval[-1] for i in (self.portfollio.evalVect(
                dateIndex(self.portfollio.loader.price, tempbegin),
                dateIndex(self.portfollio.loader.price, tempend)))]

    def plot(self):
        plt.plot(range(len(self.eval) - 1), self.eval[1:])

    def show(self):
        plt.show()

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
