import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Wczytanie danych
df = pd.read_csv('CDProject.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# # Tworzenie wykresów ACF i PACF
# fig, ax = plt.subplots(2, figsize=(12, 6))
# plot_acf(df['Close'], ax=ax[0])  # Wykres ACF
# plot_pacf(df['Close'], ax=ax[1])  # Wykres PACF
# plt.savefig('wykresy ACF i PACF')
# plt.show()

#określenie parametrów p d q domodelu arima



import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Załadowanie danych (zakładam, że masz już wczytane dane w zmiennej df)
# df = pd.read_csv("ścieżka_do_pliku.csv")

# Ustawienie częstotliwości indeksu dat na jedną jednostkę (np. dzień)
df = df.asfreq('D')

# Podział na dane treningowe i testowe
train_end = pd.to_datetime('2021-02-22')
test_start = pd.to_datetime('2021-02-23')
train_data = df['Close'][:train_end]
test_data = df['Close'][test_start:]

# Tworzenie modelu ARIMA
model = ARIMA(train_data, order=(1, 1, 1))  # p=1, d=1, q=1
model_fit = model.fit()

# Wykonanie prognozy
n_forecast = len(test_data)  # ilość kroków prognozy
#forecast, stderr, conf_int = model_fit.forecast(steps=n_forecast)
results = model_fit.forecast(steps=n_forecast)
forecast = results[0]
stderr = results[1]
conf_int = results[2]

print(forecast)
# Wykreślenie wyników
plt.figure(figsize=(10, 4))
plt.plot(train_data, color='blue', label='Training Data')
plt.plot(test_data.index, forecast, color='red', label='Forecast')
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
