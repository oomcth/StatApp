
# Charge les données des actifs souhaités depuis Yahoo finance
# Certaines données sont légérement décalés dans le temps pour palier à des soucis
# liés à des jours d'ouverture différents sur différent marché

import pandas as pd
import yfinance as yf
from data import stocks, crypto
from datetime import timedelta
from cppfct import c_isNan, dateToDatetime
import numpy as np


class Loader:

    def __init__(self, stocks, crypto, begin, end, b, e, step, drop=True) -> None:
        self.step = step  # fréquence des données. exemple 1wk = weekly
        self.price = pd.DataFrame.empty  # DataFrame final des prix
        self.vol = pd.DataFrame.empty  # DataFrame final des volumes
        self.all = stocks + crypto  # Liste des actifs étudiés

        if not(self.all == []):
            dataL = []

            for n, i in enumerate(self.all):
                # chargements des données grâce à la librairie de Yahoo finance
                df = yf.Ticker(i).history(start=b, end=e+timedelta(days=1),
                                          interval=step)
                # stoque le nom des actifs dans le dataFrame
                df['Symbol'] = i

                dataL.append(df)

                # vérifie l'intégrité des données. ie. qu'elle ne contienne pas de NaN
                if (not c_isNan(np.array(df['Close']))) or not(drop):

                    # harmonise les dates entre les différents actifs et change leur format en Date
                    df.reset_index(inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                    dataL.append(df)
                    if n != 0:
                        df['Date'] = dataL[0]['Date']
                else:

                    # si il y a une erreur dans les données, supprime l'actif du projet
                    self.all.remove(i)
            all_data = dataL[0]

            # extraction des données utiles dans les données fournis par Yahoo
            for i in range(1, len(dataL)):
                all_data = pd.concat([all_data, dataL[i]])
            self.alldata = all_data

            self.price = self.time_serie(self.alldata, "Close")
            self.vol = self.time_serie(self.alldata, "Close")

    # retourne le dataFrame finale correspondant à l'entrée name : Close / Vol par exemple
    def time_serie(self, data, name):
        df = data.reset_index()
        df = df.pivot_table(index='Date',
                            columns='Symbol',
                            values=name,
                            aggfunc='max')
        df_nindex = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df.reindex(df_nindex)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].astype("string")
        df['Date'] = df['Date'].str.split(' ').str[0]
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    # retourne le dataFrame des prix entre deux dates
    def PriceDate(self, date1, date2):
        return self.price[self.all + ['Date']].loc[(self.price['Date'] >= date1) &
                                                   (self.price['Date'] <= date2)]

    # retourne le dataFrame des volumes entre deux dates
    def VolumeDate(self, date1, date2):
        return self.vol[self.all + ['Date']].loc[(self.vol['Date'] >= date1) &
                                                 (self.vol['Date'] <= date2)]


# fonction de debugage
if __name__ == "__main__":
    end = '2022-12-30'
    e = dateToDatetime(end)
    begin = '2021-01-01'
    b = dateToDatetime(begin)
    loader = Loader(stocks, crypto, begin, end, b, e, "1wk")
    print(type(loader.price))
