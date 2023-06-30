import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych
df = pd.read_csv('CDProject.csv')

# Konwersja daty na typ daty i ustawienie jej jako indeksu
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Zapisanie 'Close' jako target
y = df['Close'].values

# Usunięcie kolumny 'Close' z danych
X = df.drop(['Close'], axis=1).values

# Skalowanie danych za pomocą MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Podział na dane treningowe i testowe w sposób sekwencyjny
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Trening modelu
model = LinearRegression()
model.fit(X_train, y_train)

# Przewidywanie na danych testowych
y_pred = model.predict(X_test)

# Tworzenie pełnej listy wartości 'Close' z przewidzianymi wartościami w odpowiednich miejscach
y_full = np.concatenate([y_train, y_pred])



from sklearn.metrics import mean_squared_error

# Obliczanie błędu średniokwadratowego
mse = mean_squared_error(y_test, y_pred)

print(f'MSE: {mse}')


from sklearn.metrics import r2_score

# Obliczanie współczynnika determinacji R^2
r2 = r2_score(y_test, y_pred)

print(f'R^2: {r2}')





# Wykres wszystkich rzeczywistych wartości 'Close' i przewidywanych wartości 'Close' na zestawie testowym
plt.figure(figsize=(10, 6))
plt.plot(df.index, y, color='blue', label='Rzeczywiste wartości Close')
plt.plot(df.index[train_size:], y_pred, color='red', label='Przewidywane wartości Close')
plt.title('Przewidywanie cen zamknięcia')
plt.xlabel('Czas')
plt.ylabel('Cena zamknięcia')
plt.legend()
plt.savefig('modelregresji.png')
plt.show()


