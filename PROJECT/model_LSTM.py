from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Wczytanie danych
df = pd.read_csv("OTGLF.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna()

# Interpolacja volume i zapis nowego pliku
df['Volume'] = df['Volume'].replace(0, np.nan)
df['Volume'] = df['Volume'].interpolate(method='linear')
df.to_csv('CDProject.csv', index=False)

df = df[['Date', 'Close']]
df = df.set_index('Date')

# Podział na dane treningowe i testowe
train_end = pd.to_datetime('2021-04-15')
test_start = pd.to_datetime('2021-04-16')

train = df[df.index <= train_end]
test = df[df.index >= test_start]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# How many past days we want to use to predict the next value
prediction_days = 100

x_train = []
y_train = []

for x in range(prediction_days, len(train_scaled)):
    x_train.append(train_scaled[x-prediction_days:x, 0])
    y_train.append(train_scaled[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Prepare test dataset
x_test = []
y_test = df.loc[test_start:, 'Close'].values[prediction_days:]

for i in range(prediction_days, len(test_scaled)):
    x_test.append(test_scaled[i-prediction_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Making Predictions


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
predictions = predictions.flatten()

# Plotting the results
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Close'], color='blue', label='Rzeczywiste wartości')
plt.plot(df.index[len(df) - len(predictions):], predictions , color='red', label='Prognozowane wartości')
plt.title('Prognoza Model LSTM')
plt.xlabel('Czas')
plt.ylabel('Kurs akcji')
plt.legend()
#plt.savefig('model-LSTM.png')
plt.show()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f'mean_absolute_error: {mae}')
print(f'mean_squared_error: {mse}')
print(f'rmse : {rmse}')
print(f'r2_score: {r2}')

