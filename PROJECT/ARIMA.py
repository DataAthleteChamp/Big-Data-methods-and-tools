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
size = int(len(df) * 0.8)
#train, test = df[0:-30], df[-30:]
train, test = df['Close'][0:size], df['Close'][size:len(df)]

# Ustalenie cechy X i etykiety y
history = [x for x in train]
predictions = list()

# Pętla po danych testowych i prognozowanie
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    pred = output[0]
    predictions.append(pred)
    obs = test[t]
    history.append(obs)


# Wykres danych i prognozy
#plt.plot(test.index, test, label='Original Data')
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





