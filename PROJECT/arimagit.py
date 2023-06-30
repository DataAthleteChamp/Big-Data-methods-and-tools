import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("CDProject.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Podział na dane treningowe i testowe
train, test = df['Close'][0:-100], df['Close'][-100:]

# Stworzenie i trenowanie modelu ARIMA
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

# Prognozowanie dla zestawu testowego
start = len(train)
end = len(train) + len(test) - 1
predictions = model_fit.predict(start=start, end=end)

# Wykres danych i prognozy
plt.plot(df.index, df['Close'], color='blue', label='Rzeczywiste wartości')
plt.plot(test.index, predictions, color='red', label='Wartość przewidywana')
plt.title('Predykcja model ARIMA')
plt.xlabel('Lata')
plt.ylabel('Cena zamknięcia')
plt.legend()
plt.savefig('ARIMA.png')
plt.show()

mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test, predictions)

print(f'mean_absolute_error: {mae}')
print(f'mean_squared_error: {mse}')
print(f'rmse: {rmse}')
print(f'r2_score: {r2}')
