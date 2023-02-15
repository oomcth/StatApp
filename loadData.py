import pandas as pd
import yfinance as yf
from data import stocks, crypto
from datetime import timedelta
from cppfct import c_isNan, contains
import numpy as np
from datetime import date


class Loader:

    def __init__(self, stocks, crypto, begin, end, b, e, step) -> None:
        self.stocks = stocks
        self.crypto = crypto
        self.step = step
        self.price = pd.DataFrame.empty
        self.all = stocks + crypto

# Stocks
        if not(self.all == []):
            dataL = []

            for i in self.all:
                df = yf.Ticker(i).history(start=b, end=e+timedelta(days=1), interval=step)
                df['Symbol'] = i
                dataL.append(df)
                if not c_isNan(np.array(df['Close'])):
                    df.reset_index(inplace=True)
                    df['Date'] = pd.to_datetime(df['Date']).dt.date
                    dataL.append(df)
                else:
                    self.all.remove(i)
            all_data = dataL[0]

            for i in range(1, len(dataL)):
                all_data = pd.concat([all_data, dataL[i]])
            self.alldata = all_data

            self.price = self.time_serie(self.alldata, "Close")
            self.vol = self.time_serie(self.alldata, "Close")




    def time_serie(self, data, name):
        df = data.reset_index()
        # df.set_index(['Date', 'Symbol'], inplace=True)
        df = df.pivot_table(index='Date', columns='Symbol', values='Close', aggfunc='max')
        df_nindex = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df.reindex(df_nindex)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].astype("string")
        df['Date'] = df['Date'].str.split(' ').str[0]
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def PriceDate(self, date1, date2):
        return self.price[self.all + ['Date']].loc[(self.price['Date'] >= date1) &
                                           (self.price['Date'] <= date2)]

    def VolumeDate(self, date1, date2):
        return self.vol[self.all + ['Date']].loc[(self.vol['Date'] >= date1) &
                                         (self.vol['Date'] <= date2)]


if __name__ == "__main__":
    end = '2022-12-30'
    e = date(2022, 12, 30)
    begin = '2021-11-11'
    b = date(2021, 11, 11)
    loader = Loader(stocks, crypto, begin, end, b, e, "1wk")
    print(loader.price)
    # print(loader.PriceDate('2022-12-26', '2022-12-30'))
