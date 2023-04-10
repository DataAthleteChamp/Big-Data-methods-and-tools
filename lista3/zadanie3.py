import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler #scikit learn

# Tworzenie syntetycznego szeregu czasowego
#np.random.seed(0)
time_series = np.random.randint(50, 276, size=10).reshape(-1, 1)

# Skalowanie szeregu czasowego do przedziału [0, 1] za pomocą MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_time_series = scaler.fit_transform(time_series)
#fit - dopasowanie- wartosci do skalowania
# transform transformacja przy uzyciu wartosci do skalowania

# Rysowanie wykresów przed i po skalowaniu
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time_series, marker="o")
plt.title("Przed skalowaniem")
plt.xlabel("Indeks")
plt.ylabel("Wartość")

plt.subplot(1, 2, 2)
plt.plot(scaled_time_series, marker="o")
plt.title("Po skalowaniu")
plt.xlabel("Indeks")
plt.ylabel("Wartość")

plt.show()
