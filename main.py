
from portfollio import Portfollio
import matplotlib.pyplot as plt
from data import stocks, crypto
from datetime import date
import pandas as pd
import datetime
from robot import Robot
from cppfct import dateIndex, dateToDatetime
import matplotlib.pyplot as plt

demo = False

end = '2022-12-30'
e = dateToDatetime(end)
begin = '2021-12-30'
b = dateToDatetime(begin)


if(demo):
    portfollio = Portfollio(begin, b, end, e, stocks, crypto, "1d")

    target1 = 1
    portfollio.optimize("maxDiv", target1)

    temp = pd.DataFrame.to_numpy(portfollio.loader.price)

    rbegin = str.split(str(temp[0][0]), " ")[0]
    rend = str.split(str(temp[-1][0]), " ")[0]

    portfollio.plot(dateIndex(portfollio.data, rbegin),
                    dateIndex(portfollio.data, rend))
    plt.show()

    f = portfollio.effiscientFrontier()
    plt.plot([i[2] for i in f], [i[1] for i in f])
    plt.show()

else:
    robot = Robot(begin, b, end, e, stocks, crypto, "1d")

    robot.setStep("3month")

    robot.setStrategie("maxDiv")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("sharpRatio")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("equalWeights")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("minVar")
    robot.RUN(begin, b, end, e)
    robot.plot()

    print(robot.portfollio.data.columns)

    plt.legend("1234")

    robot.show()
