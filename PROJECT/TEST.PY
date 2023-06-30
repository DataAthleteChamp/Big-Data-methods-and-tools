from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#przygotowanie danych
sns.set_style("darkgrid")
df = pd.read_csv("OTGLF.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna()

#interpolacja volume zapis nowego pliku
df['Volume'] = df['Volume'].replace(0, np.nan)
df['Volume'] = df['Volume'].interpolate(method='linear')
df.to_csv('CDProject.csv', index=False)

df = df[['Close']]
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# How many past days we want to use to predict the next value
prediction_days = 30

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

x_test = []
y_test = []

for i in range(prediction_days, len(test_scaled)):
    x_test.append(test_scaled[i-prediction_days:i, 0])
    y_test.append(test.values[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Making Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plotting the results
plt.figure(figsize=(10,5))
plt.plot(y_test, color='blue', label='Rzeczywiste wartości')
plt.plot(predictions , color='red', label='Prognozowane wartości')
plt.title('Prognoza')
plt.xlabel('Czas')
plt.ylabel('Kurs akcji')
plt.legend()
plt.savefig('LSTM modell')
plt.show()

#error = mean_squared_error(test[prediction_days:], predictions)

# #error = mean_squared_error(test, predictions)
# print(f'Błąd średniokwadratowy: {error}')

mae = mean_absolute_error(test[prediction_days:], predictions)
mse = mean_squared_error(test[prediction_days:], predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test[prediction_days:], predictions)

print(f'mean_absolute_error: {mae}')
print(f'mean_squared_error: {mse}')
print(f'rmse : {rmse}')
print(f'r2_score: {r2}')



""""
osobno modele w osobnych plikach



"""