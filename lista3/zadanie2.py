import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #scikit learn

# Tworzenie przykładowego szeregu czasowego
#time_series = np.array([50, 75, 100, 125, 150, 175, 200, 225, 250, 275]).reshape(-1, 1)
time_series = np.array([50, 75, 125, 200, 250, 175, 100, 225, 150, 275]).reshape(-1, 1)

# Standaryzacja szeregu czasowego
scaler = StandardScaler()
standardized_time_series = scaler.fit_transform(time_series) #metoda

# Rysowanie wykresów przed i po standaryzacji
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time_series, marker="o")
plt.title("Przed standaryzacją")
plt.xlabel("Indeks")
plt.ylabel("Wartość")

plt.subplot(1, 2, 2)
plt.plot(standardized_time_series, marker="o")
plt.title("Po standaryzacji")
plt.xlabel("Indeks")
plt.ylabel("Wartość")

plt.show()


#Standaryzacja jest techniką przekształcania danych,
# która polega na zmianie ich średniej wartości na 0 i odchylenia standardowego na 1 -> rozkład normalny