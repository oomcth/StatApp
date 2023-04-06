
# fonction utilise dans le reste du code
# certaine sont précompilée en c pour améliorer les performances


import numba
import numpy as np
from math import isnan
import datetime


# calcul la contribution à la variance d'un actif
@numba.jit(nopython=True)
def c_Variancecontrib(w1, weights, cov):
    return (np.transpose(w1) @ (cov @
            weights) /
            (np.transpose(weights) @ (cov) @ weights))


# converti un dateTime en Date
def dateTimeToDate(date):
    return str(date.year) + "-" + str(date.month) + "-" + str(date.day)


# converti un Date en dateTime

def dateToDatetime(date):
    temp = str.split(date, " ")[0]
    temp2 = str.split(temp, "-")
    return datetime.date(int(temp2[0]), int(temp2[1]), int(temp2[2]))


# renvoie s'il existe un NaN dans un vecteur
def c_isNan(vect):
    for i in vect:
        if isnan(i):
            return True
        return False


# renvoie si un élément est dans une array
def contains(arr, el):
    for i in arr:
        if i == el:
            return True
    return False


# renvoie l'index dans le dataframe de la date la plus proche par excès de la date fournis en entrée
def dateIndex(df, date, t=0):
    if t == 14:
        raise "error +14"
    if not(list((df[df['Date'] == date].index.values)) == []):
        return df[df['Date'] == date].index.values
    else:
        return dateIndex(df, strnextd(date), t+1)


# renvoie un Date correspondant au lendemain d'une date en format string
def strnextd(s):
    d = datetime.date(int(str.split(str(s), "-")[0]),
                int(str.split(s, "-")[1]),
                int(str.split(s, "-")[2])) + datetime.timedelta(days=1)
    return dateTimeToDate(d)


# renvoie la date au format string additionné d'un certain nombre de jours
def datestringdiff(date, days):
    temp = dateToDatetime(date)
    temp += datetime.timedelta(days=days)
    return dateTimeToDate(temp)
