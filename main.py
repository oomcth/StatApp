
from portfollio import Portfollio
import matplotlib.pyplot as plt
from data import stocks, crypto
import pandas as pd
from robot import Robot
from cppfct import dateIndex, dateToDatetime


demo = True

end = '2022-12-30'
e = dateToDatetime(end)
begin = '2018-01-01'
b = dateToDatetime(begin)


if(demo):
    portfollio = Portfollio(begin, b, end, e, stocks, crypto, "1wk")

    target1 = 0.001
    portfollio.optimize("equalWeights", 0.1)

    temp = pd.DataFrame.to_numpy(portfollio.loader.price)

    rbegin = str.split(str(temp[0][0]), " ")[0]
    rend = str.split(str(temp[-1][0]), " ")[0]

    portfollio.plot(dateIndex(portfollio.data, rbegin),
                    dateIndex(portfollio.data, rend))
    portfollio.info()
    plt.show()
    portfollio.plotAll()

else:
    robot = Robot(begin, b, end, e, stocks, crypto, "1wk")

    robot.setStep("3month")

    robot.setStrategie("sharpRatio")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("equalRisk")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("equalWeights")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("minVar")
    robot.RUN(begin, b, end, e)
    robot.plot()

    robot.setStrategie("maxDiv")
    robot.RUN(begin, b, end, e)
    robot.plot()

    print(robot.portfollio.data.columns)

    plt.legend("12345")

    robot.show()
