from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("CDProject.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# teraz możemy stworzyć kolumnę 'DateNumeric'
df['DateNumeric'] = (df.index - df.index[0]).days

# Ustalenie cechy X i etykiety y
X = df['DateNumeric'].values.reshape(-1,1)
y = df['Close'].values

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Tworzenie i uczenie modelu regresji liniowej
model = LinearRegression()
model.fit(X_train, y_train)

# Przewidywanie na danych testowych
y_pred = model.predict(X_test)

# Wykres danych i predykcji
plt.figure(figsize=(10, 6))
plt.plot(X, y, color='blue', label='Wartości rzeczywiste')
#plt.plot(X_test, y_test, color='blue', label='Wartości rzeczywiste')
plt.plot(X_test, y_pred, color='red', label='Regresja liniowa')
plt.title('Regresja liniowa na danych giełdowych CD Project')
plt.xlabel('Liczba dni od początku okresu')
plt.ylabel('Cena zamknięcia')
plt.legend()
plt.savefig('regresja_liniowa.png')
plt.show()


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'mean_absolute_error: {mae}')
print(f'mean_squared_error: {mse}')
print(f'rmse: {rmse}')
print(f'r2_score: {r2}')
