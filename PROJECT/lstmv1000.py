from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import talib

# Collecting stock data
stock = yf.Ticker("AAPL")
data = stock.history(period="max")

# Calculate technical indicators
data['RSI'] = talib.RSI(data['Close'].values, timeperiod=14)
data['SMA'] = talib.SMA(data['Close'].values, timeperiod=30)
data['EMA'] = talib.EMA(data['Close'].values, timeperiod=30)
data['ADX'] = talib.ADX(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
data['CCI'] = talib.CCI(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)
data['ROC'] = talib.ROC(data['Close'].values, timeperiod=10)

# Drop NaN
data.dropna(inplace=True)

# Split into train and test sets
train_size = int(len(data) * 0.67)
train, test = data[0:train_size], data[train_size:len(data)]
print(len(train), len(test))

# Convert DataFrame into array
train = train.values
test = test.values

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Create separate scaler for 'Close' price
close_scaler = MinMaxScaler()
close_scaler.fit(train[:, :1])

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t and Y=t+1
look_back = 1
x_train, y_train = create_dataset(train, look_back)
x_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=25, batch_size=1, verbose=2)

# Make predictions
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)

# Inverse transform predictions and y values
trainPredict = close_scaler.inverse_transform(trainPredict)
y_train = close_scaler.inverse_transform([y_train])
testPredict = close_scaler.inverse_transform(testPredict)
y_test = close_scaler.inverse_transform([y_test])

# Calculate error scores
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
