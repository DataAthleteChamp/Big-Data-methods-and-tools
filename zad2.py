from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Generowanie danych
x = np.linspace(-3, 3, 100)  # Generujemy 100 równoodległych punktów od -3 do 3 i zapisujemy je w tablicy x
szum = np.random.normal(0, 1, size=100)  # Tworzymy szum losowy z rozkładu normalnego
y = 2 + x + 0.5*x**2 + szum  # Tworzymy wartości y jako wartości funkcji kwadratowej z dodanym szumem losowym

# Rozszerzenie x na n-wymiarową macierz
poly = PolynomialFeatures(degree=2)  # Tworzymy instancję klasy PolynomialFeatures, która pozwoli nam rozszerzyć tablicę x na macierz o wymiarach (100, 2) zawierającą kolumny x oraz x^2
X_poly = poly.fit_transform(x.reshape(-1, 1))  # Rozszerzamy tablicę x na macierz z użyciem metody fit_transform()

# Tworzenie modelu regresji wielomianowej
model = linear_model.LinearRegression()  # Tworzymy instancję klasy LinearRegression, która reprezentuje model regresji liniowej
model.fit(X_poly, y)  # Dopasowujemy model do danych, używając metody fit(). W tym przypadku używamy macierzy X_poly jako danych wejściowych, a tablicy y jako zmienną zależną.

# Ocena modelu na danych treningowych
y_pred = model.predict(X_poly)  # Wykonujemy predykcję dla danych treningowych
# mse = metrics.mean_squared_error(y, y_pred)  # Obliczamy błąd średniokwadratowy
# r2 = metrics.r2_score(y, y_pred)  # Obliczamy współczynnik determinacji

# Rysowanie wykresu z punktami i dopasowaną krzywą
plt.scatter(x, y)  # Rysujemy punkty na wykresie, używając funkcji scatter() z biblioteki matplotlib
plt.plot(x, y_pred, color='red')  # Rysujemy dopasowaną krzywą na wykresie, używając funkcji plot() z biblioteki matplotlib
# plt.title(f'Degree = {poly.degree}, MSE = {mse:.2f}, R2 = {r2:.2f}')  # Dodajemy tytuł wykresu z informacją o stopniu wielomianu, błędzie średniokwadratowym i współczynniku determinacji
plt.show()  # Wyświetlamy wykres

#
# lista kroków:
# generuj dane
# dodaj szum
# zaproponuj model funkcje
# Rozszerzamy tablicę x na macierz zawierającą kolumny x oraz x^2
# Tworzymy instancję modelu regresji liniowej i dopasowujemy ją do danych
# Dokonujemy predykcji i obliczamy błąd średniokwadratowy oraz współczynnik determinacji
# Rysujemy wykres z punktami i dopasowaną krzywą



#
# START
# ├─ importuj biblioteki
# ├─ wygeneruj dane
# │  ├─ wygeneruj tablicę x
# │  ├─ wygeneruj szum losowy
# │  └─ wygeneruj tablicę y jako wartości funkcji kwadratowej z dodanym szumem losowym
# ├─ rozszerz tablicę x na macierz
# │  ├─ stwórz instancję klasy PolynomialFeatures
# │  └─ rozszerz tablicę x na macierz z użyciem metody fit_transform()
# ├─ dopasuj model regresji wielomianowej do danych
# │  ├─ stwórz instancję klasy LinearRegression
# │  └─ dopasuj model do danych z użyciem metody fit()
# ├─ dokonaj predykcji na danych treningowych
# │  └─ użyj metody predict() na dopasowanym modelu
# ├─ oblicz błąd średniokwadratowy i współczynnik determinacji
# │  ├─ oblicz MSE z użyciem funkcji mean_squared_error()
# │  └─ oblicz R2 z użyciem funkcji r2_score()
# ├─ narysuj wykres z punktami i dopasowaną krzywą
# │  ├─ narysuj punkty na wykresie z użyciem funkcji scatter()
# │  ├─ narysuj dopasowaną krzywą na wykresie z użyciem funkcji plot()
# │  ├─ dodaj tytuł wykresu z informacją o stopniu wielomianu, błędzie średniokwadratowym i współczynniku determinacji
# │  └─ wyświetl wykres z użyciem funkcji show()
# KONIEC









