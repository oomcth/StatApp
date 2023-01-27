import pandas as pd
import yfinance as yf
from data import stocks
from datetime import timedelta


class Loader:

    def __init__(self, stocks, begin, end, b, e) -> None:
        self.stocks = stocks

        dataL = []
        for i in self.stocks:
            df = yf.Ticker(i).history(start=b, end=e+timedelta(days=1))
            df['Symbol'] = i
            if (e-b).days + 1 == len(df):
                dataL.append(df)
            else:
                self.stocks.remove(i)
        all_data = dataL[0]

        for i in range(1, len(dataL)):
            all_data = pd.concat([all_data, dataL[i]])
        self.alldata = all_data
        self.price = self.time_serie(self.alldata, "Close")
        self.vol = self.time_serie(self.alldata, "Volume")

    def time_serie(self, data, name):
        df = data.reset_index()
        df.set_index(['Date', 'Symbol'], inplace=True)
        df = df[name].unstack()
        df_nindex = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df.reindex(df_nindex)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].astype("string")
        df['Date'] = df['Date'].str.split(' ').str[0]
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def PriceDate(self, date1, date2):
        print(self.stocks)
        return self.price[self.stocks].loc[(self.price['Date'] >= date1) &
                                           (self.price['Date'] <= date2)]

    def VolumeDate(self, date1, date2):
        return self.vol[self.stocks].loc[(self.vol['Date'] >= date1) &
                                         (self.vol['Date'] <= date2)]


if __name__ == "__main__":
    loader = Loader(stocks)
    print(loader.price)
    print(loader.PriceDate('2022-12-26', '2022-12-30'))
