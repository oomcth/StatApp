import yfinance as yf
from data import stocks, crypto
from portfollio import Portfollio
from cppfct import dateToDatetime, dateIndex
from datetime import timedelta


end = '2022-12-30'
e = dateToDatetime(end)
begin = '2018-01-01'
b = dateToDatetime(begin)

df = yf.Ticker("EURUSD=X").history(start=b, end=e+timedelta(days=1), interval="1wk")

print(0.85**(1/300)-1)
