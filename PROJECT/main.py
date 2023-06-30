import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA

sns.set_style("darkgrid")
df = pd.read_csv("OTGLF.csv")

# konwersja daty na odpowiedni format
df['Date'] = pd.to_datetime(df['Date'])

# print(df.head())

# df2 = df.copy(deep=True)
# df2.index = df2['Date']
# df2.drop(['Date'], inplace=True, axis=1)
# df2.plot(subplots=True, figsize=(10,6))
# # plt.savefig('wykres1')
# plt.show()


# sprawdzenie brakujących danych
# print(df.isnull().sum())

# usuwanie wierszy z brakującymi danymi (jeśli są)
df = df.dropna()

# interpolacja wartości 0 w kolumnie 'Volume'
df['Volume'] = df['Volume'].replace(0, np.nan)
df['Volume'] = df['Volume'].interpolate(method='linear')


# zapis DataFrame do pliku CSV
df.to_csv('CDProject.csv', index=False)


# wybieramy 'Close' jako naszą cechę do prognozowania
df = df[['Close']]

# 80% dla treningu, 20% dla testu
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:len(df)]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# # trenowanie modelu regresji liniowej
# model = LinearRegression()
# model.fit(train_scaled, np.arange(len(train_scaled)))  # Model jest teraz trenowany na danych treningowych
#
# # prognozowanie na podstawie modelu
# predictions = model.predict(test_scaled)
#
#
# # obliczanie błędu średniokwadratowego
# error = mean_squared_error(test_scaled, predictions)
# print(f'Błąd średniokwadratowy: {error}')
#
#
# # plotowanie prawdziwych i prognozowanych wartości
# plt.figure(figsize=(12,6))
# plt.plot(df.index, df['Close'], color='blue', label='Rzeczywiste wartości')
# plt.plot(test.index, predictions, color='red', label='Prognozowane wartości')
# plt.title('Prognozowanie kursu akcji')
# plt.xlabel('Data')
# plt.ylabel('Kurs akcji')
# plt.legend()
# plt.savefig('predykcja regresja liniowa- nieudana')
# plt.show()

# ARIMA model
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Prognozowanie
#predictions, stderr, conf_int = model_fit.forecast(steps=len(test))
#predictions, stderr = model_fit.forecast(steps=len(test))
predictions = model_fit.forecast(steps=len(test))

# Obliczanie błędu średniokwadratowego
error = mean_squared_error(test, predictions)
print(f'Błąd średniokwadratowy: {error}')

# Plotowanie prawdziwych i prognozowanych wartości
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], color='blue', label='Rzeczywiste wartości')
plt.plot(test.index, predictions, color='red', label='Prognozowane wartości')
plt.title('Prognozowanie kursu akcji')
plt.xlabel('Data')
plt.ylabel('Kurs akcji')
plt.legend()
plt.savefig('ARIMA model')
plt.show()