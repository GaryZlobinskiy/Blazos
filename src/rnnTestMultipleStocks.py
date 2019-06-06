#https://www.marketwatch.com/game/stockpredictionrnn
#UNUSED RNN


from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

key = "PQ5WZMLZQN3MBGHC"
stockList = ["WMT","RDS.A","TM","BP","XOM","BRK.B","AAPL","MCK","UNH","CVS","AMZN","T","GM","F","ABC","TOT","HMC","CVX","CAH","COST","VZ","KR","GE","LFC","WBA","BNPQY","BACHY","JPM","FNMA","OGZPY","PRU","BMWYY","GOOG","NSANY","NTTYY","HD","ARZGF","BAC","WFC","LUKOY","BA","DNFGF","SIEGY","PSX","CRRFY","ANTM","MSFT","VLO","C","SAN","HYMTF","UAL","DAL","LUV","JBLU","AAL","BDRBF","GS","DJI","FTSE","IXIC","GSPC","SNAP","SBUX","TSLA","TWTR","FB","UBER","NYSE"]

class RNN():
    def __init__(self, look_back, X_train):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(60, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
        self.model.add(tf.keras.layers.LSTM(25))
        self.model.add(tf.keras.layers.Dense(5))
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

        self.history = None

    def call_model(self, X_train, X_test, y_train, y_test):
        print("Creating model...")

        print("Training...")
        self.history = self.model.fit(X_train, y_train, epochs=2, batch_size=1, verbose=1, validation_data=(X_test, y_test))

        print("Saving...")
        self.model.save_weights("rnn_model.h5")

        print("Plotting...")
        plt.figure(figsize=(16, 10))
        plt.plot(self.history.epoch, self.history.history["val_mean_absolute_error"], label="Validation")
        plt.plot(self.history.epoch, self.history.history["mean_absolute_error"], label="Train")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.show()



def main():
    rnn = None

    for stock in stockList:
        ts = TimeSeries(key, output_format="pandas")
        data, _ = ts.get_daily(symbol=stock, outputsize="full")
        pprint(data.head(2))

        data.to_csv("rnn_test_data.csv")

        print("Reading data...")
        data = pd.read_csv("rnn_test_data.csv")
        data.index = data.date
        data.sort_index(ascending=True, axis=0)
        data.drop("date", axis=1, inplace=True)

        print("Scaling data...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(data.values)
        look_back = 60
        print("Processing data...")
        X = []
        y = []
        for i in range(look_back, len(data)):
            X.append(np.array(values[i - look_back:i]))
            y.append(np.array(values[i]))

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, train_size=0.8)
        print(X_train.shape)

        if (rnn == None):
            rnn = RNN(look_back, X_train)

        X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
        X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))
        y_train = np.reshape(y_train, (y_train.shape[0], 5))
        y_test = np.reshape(y_test, (y_test.shape[0], 5))

        rnn.call_model(X_train, X_test, y_train, y_test)

    print("Evaluating...")
    test_loss, test_acc = rnn.model.evaluate(X_test, y_test)
    print("Test accuracy: ", test_acc)


if __name__ == "__main__":
    main()