import numba
import numpy as np
from math import isnan
import datetime


@numba.jit(nopython=True)
def c_Variancecontrib(w1, weights, cov):
    return (np.transpose(w1) @ (cov @
            weights) /
            (np.transpose(weights) @ (cov) @ weights))


def dateTimeToDate(date):
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)

def dateToDatetime(date):
    temp = str.split(date, " ")[0]
    temp2 = str.split(temp, "-")
    return datetime.date(int(temp2[0]), int(temp2[1]), int(temp2[2]))

@numba.njit
def c_isNan(vect):
    for i in vect:
        if isnan(i):
            return True
        return False


def dateIndex(df, date, t=0):
    if t == 4:
        raise "error"
    if not(list((df[df['Date'] == date].index.values)) == []):
        return df[df['Date'] == date].index.values
    else:
        return dateIndex(df, strnextd(date), t+1)


def strnextd(s):
    d = datetime.date(int(str.split(str(s), "-")[0]),
                int(str.split(s, "-")[1]),
                int(str.split(s, "-")[2])) + datetime.timedelta(days=1)
    return dateTimeToDate(d)
