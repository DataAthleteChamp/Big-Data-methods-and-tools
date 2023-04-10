import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler #scikit learn

# Tworzenie przykładowego szeregu czasowego
time_series = np.array([50, 75, 125, 200, 250, 175, 100, 225, 150, 275]).reshape(-1, 1)
#reshape wymagny przez klase minmax scaller macierz 10 na 1

# Normalizacja szeregu czasowego do przedziału [0, 1] za pomocą MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_time_series = scaler.fit_transform(time_series)

# Rysowanie wykresów przed i po normalizacji
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time_series, marker="o")
plt.title("Przed normalizacją")
plt.xlabel("Indeks")
plt.ylabel("Wartość")

plt.subplot(1, 2, 2)
plt.plot(normalized_time_series, marker="o")
plt.title("Po normalizacji")
plt.xlabel("Indeks")
plt.ylabel("Wartość")

plt.show()

#Szereg czasowy to sekwencja punktów danych, które są zebrane lub zarejestrowane w równomiernych odstępach czasu.
#Normalizacja to proces przekształcania danych do wspólnego zakresu lub skali

# Funkcja fit_transform() klasy MinMaxScaler wykonuje dwie operacje:
# Wylicza wartości minimalne i maksymalne dla każdej cechy w zbiorze danych uczących.
# Przeprowadza skalowanie wartości każdej cechy z przedziału [min, max] na przedział [a, b].

# Klasa MinMaxScaler działa według następującego algorytmu:
# Znajduje wartość minimalną i maksymalną dla każdej cechy w zbiorze danych uczących.
# Przeprowadza skalowanie wartości każdej cechy z przedziału [min, max] na przedział [a, b], gdzie a i b to wartości minimalna i maksymalna w przedziale docelowym (domyślnie [0, 1] lub [-1, 1] w zależności od wartości argumentu feature_range).
# Zapisuje parametry skalowania, aby można je było wykorzystać do transformacji nowych danych testowych.