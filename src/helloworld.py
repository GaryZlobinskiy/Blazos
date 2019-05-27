stock = input("What Stock Would you like to invest in? (Write the stock marker; Ex; MSFT = microsoft\n")

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
ts = TimeSeries(key='PQ5WZMLZQN3MBGHC', output_format='pandas')
data, metadata = ts.get_intraday(symbol=stock, interval='60min', outputsize='full')
pprint(data.head(90000))
data.to_csv('StockInfoFinal.csv', sep=',')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('StockInfoFinal.csv')
X = dataset.iloc[:, 1:7].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mse'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred == y)

new_prediction = classifier.predict(sc.transform(np.array([[100,100,100,100]])))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)