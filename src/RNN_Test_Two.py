#help from https://towardsdatascience.com/predicting-stock-price-with-lstm-13af86a74944

from keras.models import Sequential, load_model
import os
from keras.layers import Dense
import time
import sys
from keras.layers import Dropout
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from keras.callbacks import CSVLogger
from keras.layers import LSTM
import logging
from keras import optimizers
from tdqm import tqdm_notebook

PATH = 'C:\\Users\\garyz\\PycharmProjects\\Blazos\\src\\testData.csv'
BATCH_SIZE = 3
TIME_STEPS = 60
y_col_index = 3

data = pd.read_csv(PATH)
data.index = data.date
data.sort_index(ascending=True, axis=0)
data.drop("date", axis=1, inplace=True)

#plotting MSFT data

'''
plt.figure()
plt.plot(data["1. open"])
plt.plot(data["2. high"])
plt.plot(data["3. low"])
plt.plot(data["4. close"])
plt.title('MSFT stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()

plt.figure()
plt.plot(data["5. volume"])
plt.title('MSFT stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()
'''

print("checking if any null values are present\n", data.isna().sum())

train_columns = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(data_train), len(data_test))

#scale the feature MixMax, build array
x = data_train.loc[:, train_columns].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(data_test.loc[:, train_columns])

def build_timeseries(mat, y_col_index):
    # y_col_index is the index of the column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros(dim_0,)

    for i in tdqm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+1, y_col_index]
    print("length of time-series i/o", x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    '''
     trims dataset to a size that's divisible by BATCH_SIZE
    '''
    no_of_row_drop = mat.shape[0]%batch_size
    if(no_of_row_drop > 0):
        return mat[:-no_of_row_drop]
    else:
        return mat

x_t, y_t = build_timeseries(x_train, 3)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = build_timeseries(x_test, 3)
x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE), 2)
y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE), 2)

lstm_model = Sequential()
lstm_model.add(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform')
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20, activation='relu'))
lstm_model.add(Dense(1, activation='linear')) #could use sigmoid
optimizer = optimizers.RMSprop(lr='lr')
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

csv_logger = CSVLogger(os.path.join("rnn_model2.h5", 'RNN_Test_Two' + '.log'), append=True)

history = lstm_model.fit(x_t, y_t, epochs=2, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                    trim_dataset(y_val, BATCH_SIZE)),
                    callbacks=[csv_logger])
