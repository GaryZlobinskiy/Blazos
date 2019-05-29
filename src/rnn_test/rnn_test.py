import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("Reading data...")
data = pd.read_csv("rnn_test_data.csv")
data.index = data.date
data.sort_index(ascending=True, axis=0)
data.drop("date", axis=1, inplace=True)

print("Scaling data...")
scaler = MinMaxScaler(feature_range=(0,1))
values = scaler.fit_transform(data.values)

look_back = 60

print("Processing data...")
X = []
y = []
for i in range(look_back, len(data)):
    X.append(np.array(values[i - look_back:i]))
    y.append(np.array(values[i][3]))

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, train_size=0.8)
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], look_back, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], look_back, X_test.shape[2]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

print("Creating model...")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

print("Training...")
history = model.fit(X_train, y_train, epochs=2, batch_size=1, verbose=1, validation_data=(X_test, y_test))

print("Plotting...")
plt.figure(figsize=(16, 10))
plt.plot(history.epoch, history.history["val_mean_squared_error"], label="Validation")
plt.plot(history.epoch, history.history["mean_squared_error"], label="Train")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()

print("Evaluating...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy: ", test_acc)