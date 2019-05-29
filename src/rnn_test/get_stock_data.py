from alpha_vantage.timeseries import TimeSeries
from pprint import pprint

stock = "MSFT"
key = "PQ5WZMLZQN3MBGHC"

ts = TimeSeries(key, output_format="pandas")
data, _ = ts.get_daily(symbol=stock, outputsize="full")
pprint(data.head(2))
data.to_csv("rnn_test_data.csv")