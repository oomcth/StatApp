import yfinance as yf
from data import stocks, crypto
from portfollio import Portfollio
from cppfct import dateToDatetime, dateIndex
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt


end = "2017-10-16"
e = dateToDatetime(end)
begin = "2017-01-16"
b = dateToDatetime(begin)

temp = Portfollio(begin, b, end, e, ['EURUSD=X'], [], "1d")
t = [1,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0]
m = np.array([[0, 0.45, 0, 0, 0.3666666667, 0, 0, 0, 0, 0.05, 0.05, 0.03333333333, 0.05, 0],
              [0, 0.55, 0, 0.03333333333, 0.3666666667, 0, 0.01666666667, 0, 0, 0, 0, 0, 0.03333333333, 0],
              [0, 0.4726791432, 0, 0.03333333333, 0.4606541902, 0, 0, 0, 0, 0, 0, 0, 0.03333333333, 0],
              [0, 0.55, 0, 0.05601222999, 0.24398777, 0, 0.06666666667, 0, 0.03333333333, 0.025, 0.025, 0, 0, 0]])

dates = ['2017-01-16',
         '2017-02-13',
         '2017-03-20',
         '2017-05-01',
         '2017-05-25']

names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']  # 'EuroStoxx Large', 'MSCI US', 'TOPIX', 'Euro MTS Global', 'US Govt Bonds All mat', 'Pan-European Aggregate Corporate', 'US Aggregate Corporate	Barclays Euro Fin Senior', 'Pan-European High Yield	€ Financials Subordinated', 'Europe Convertible Bonds', 'Inflation Euro All mat	Eonia Capi 5D']

temp.all += names

for name in names:
    print(name)
    temp.loader.addExcel("Classeur1.xlsx", name)

val = [100]

for i in range(len(dates) - 1):
    temp.ChangeWindow(dates[i], dates[i+1])
    print(temp.data)
    start = val[-1]
    val += [a * start for a in temp.evalVect(dateIndex(temp.data, dates[i]),
                                                   dateIndex(temp.data, dates[i+1]),
                                                   m[i])]

print(val[1:])



plt.plot(range(len(val[1:-1])), val[1:-1])
plt.show()
