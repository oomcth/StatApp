from loadData import Loader
from portfollio import Portfollio
import matplotlib.pyplot as plt
from data import stocks
from datetime import date
import pandas as pd

end = '2022-12-30'
e = date(2022, 12, 30)
begin = '2021-11-11'
b = date(2021, 11, 11)


def dateIndex(df, date):
    return df[df['Date'] == date].index.values


temp = Loader(stocks, begin, end, b, e)
portfollio = Portfollio(temp.PriceDate(begin, end).iloc[::-1],
                        len(temp.price.columns) - 1)

target1 = 1
portfollio.optimize("sharpRatio", target1)

portfollio.plot(dateIndex(temp.price, begin), dateIndex(temp.price, end))
plt.show()

# f = portfollio.effiscientFrontier()
# plt.plot([i[2] for i in f], [i[1] for i in f])
# plt.show()
