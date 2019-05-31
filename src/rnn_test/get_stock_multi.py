from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
from os import path
import json
import time

keys = ["PQ5WZMLZQN3MBGHC", "QQS86XEKKBN7C828", "HPZ6OOWU6AD3RKW4", "C2K3U5N3MG9NB9U6", "X8L8C4EJD5B41NSD", "SHCBRC78908O3AL3"]
wait_time = 3

f = open(path.normpath(path.join(path.dirname(__file__), "../symbols.json")))
stocks = json.load(f)

ts = [TimeSeries(key, output_format="pandas") for key in keys]

i = 0
for stock in stocks:
    print("Fetching %s" % stock)
    try:
        data, _ = ts[i % len(ts)].get_daily(symbol=stock, outputsize="full")
        pprint(data.head(2))
        data.to_csv("data/%s.csv" % stock)
    except:
        print("Failed to fetch data for %s" % stock)
    finally:
        print("Done with %s" % stock)
        time.sleep(wait_time)
        i = i + 1