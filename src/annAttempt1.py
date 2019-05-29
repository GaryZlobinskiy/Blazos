from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os


def loadStockInfo(inputPath):
    cols = ["date","open", "high", "low", "close", "volume"]
    df = pd.read_csv(inputPath, sep=",", header=None, names=cols)
    return df

def process_house_attributes(df, train, test):
    continuous = ["open", "high", "low", "close", "volume"]

    cs = MinMaxScaler()

    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    trainX = np.hstack([trainContinuous])
    testX = np.hstack([testContinuous])

    return (trainX, testX)

from keras.models import Sequential
from keras.layers.core import Dense


def create_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    if regress:
        model.add(Dense(1, activation="linear"))

    return model

inputPath = os.path.sep.join([args["dataset"], "StockInfoFinal.txt"])
df = datasets.loadStockInfo(nputPath)

(train, test) = train_test_split(df, test_size=0.2, random_state=0)

model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testX)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
    locale.currency(df["price"].mean(), grouping=True),
    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))