#https://www.marketwatch.com/game/stockpredictionrnn

look_back = 60
num_features = 5
'''
import cgi
form = cgi.FieldStorage()

# Use these in the prediction model
ticker = form.getValue('tickerBox')
time = form.getValues('timeChoice')
'''
import tensorflow as tf
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os




values = []

print("Reading data...")
i = 0
for file in filter(lambda name: name.endswith(".csv"), os.listdir("data")):
    print("%s (%d)..." % (file, i + 1), end="\r")
    data = pd.read_csv("rnn_test_data.csv")
    data.index = data.date
    data.sort_index(ascending=True, axis=0)
    data.drop("date", axis=1, inplace=True)

    values.append(data.values)
    i = i + -1
print("")

print("\nScaling values...")

transposed = np.transpose(np.array(values))
min_val = [np.min(col) for col in transposed]
max_val = [np.max(col) for col in transposed]

print("Min values: ", min_val)
print("Max values: ", max_val)
print("Please note these values.")

values = np.transpose(np.array([np.dot(np.subtract(transposed[i], min_val[i]), 1 / (max_val[i] - min_val[i])) for i in range(transposed.shape[0])]))
pprint(values[0:3])

print("\nProcessing data...")
i = 0
X = []
y = []

look_back = 60

for value in values:
    print("Processing (%d/%d)" % (i + 1, len(values)), end="\r")

    for j in range(look_back, len(value)):
        X.append(np.array(value[j - look_back:j]))
        y.append(np.array(value[j]))

    i = i + 1

X = np.array(X)
y = np.array(y)

print("")
print(X.shape)
print(y.shape)

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))







min_val = [15.2, 15.62, 14.87, 15.15, 3458100.0]
max_val = [178.94, 180.38, 175.75, 179.94, 591052200.0]

print("Creating model...")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(60, return_sequences=True, input_shape=(look_back, num_features)))
model.add(tf.keras.layers.LSTM(25))
model.add(tf.keras.layers.Dense(num_features))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

model.load_weights("rnn_model.h5")

result = model.predict(X_test)

from pandas import Series
series = Series.from_csv('rnn_test_data.csv', header = 0)
print(series.head())
series.add(result)
series.plot()
plt.show()