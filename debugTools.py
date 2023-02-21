import sys
from portfollio import Portfollio
import matplotlib.pyplot as plt
from cppfct import dateToDatetime, dateIndex
import pandas as pd

if sys.argv[1] != 'help':

    assets = sys.argv[1].split(',')

    end = str(sys.argv[3])
    e = dateToDatetime(end)
    begin = str(sys.argv[2])
    b = dateToDatetime(begin)

    portfollio = Portfollio(begin, b, end, e, assets, [], "1wk")

    target1 = 1
    portfollio.optimize(str(sys.argv[4]), target1)
    print(portfollio.weights, sum(portfollio.weights))

    temp = pd.DataFrame.to_numpy(portfollio.loader.price)

    rbegin = str.split(str(temp[0][0]), " ")[0]
    rend = str.split(str(temp[-1][0]), " ")[0]

    portfollio.plot(dateIndex(portfollio.data, rbegin),
                    dateIndex(portfollio.data, rend))
    plt.show()

    portfollio.info()

    f = portfollio.effiscientFrontier()
    plt.plot([i[2] for i in f], [i[1] for i in f])
    plt.show()
else:
    print("input 1 : assets array : asset1,asset2,asset3,... No spaces")
    print("input 2 : begin of time window : y-m-d")
    print("input 3 : end of time windiw : y-m-d")
    print("input 4 : portfollio strategy")
    print("strategies are : {sharpRatio, equalRisk, equalWeights, maxDiv, minVar}")
    