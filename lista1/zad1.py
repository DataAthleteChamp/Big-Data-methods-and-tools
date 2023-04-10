import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generowanie danych
x = np.linspace(0, 10, 100)  # Generujemy 100 równoodległych punktów od 0 do 10 i zapisujemy je w tablicy x
szum = np.random.normal(0, 1, size=100) #szum z rozkładu normalnego
y = 3*x + szum  # Tworzymy wartości y jako wartości funkcji liniowej

# Tworzenie modelu regresji liniowej
model = LinearRegression()  # Tworzymy instancję klasy LinearRegression, która reprezentuje model regresji liniowej
model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # Dopasowujemy model do danych, używając metody fit().
# W tym przypadku używamy tablicy x jako zmienną niezależną, a tablicy y jako zmienną zależną.

# Estymacja parametrów modelu
a = model.coef_[0][0]  # Wyciągamy współczynnik kierunkowy a z modelu regresji, który jest przechowywany w atrybucie coef_
b = model.intercept_[0]  # Wyciągamy wyraz wolny b z modelu regresji, który jest przechowywany w atrybucie intercept_

# Wykres z punktami i dopasowaną linią
plt.scatter(x, y)  # Rysujemy punkty na wykresie, używając funkcji scatter() z biblioteki matplotlib
plt.plot(x, a * x + b, color='red')  # Rysujemy dopasowaną linię na wykresie, używając funkcji plot() z biblioteki matplotlib.
# Linia ta ma równanie y = ax + b, gdzie a to współczynnik kierunkowy, a b to wyraz wolny.
plt.show()  # Wyświetlamy wykres






# lista kroków
# Import bibliotek numpy, matplotlib.pyplot oraz LinearRegression z biblioteki sklearn.linear_model.
# Wygeneruj 100 równoodległych punktów od 0 do 10 i zapisz je w tablicy x.
# Wygeneruj wartości y jako wartości funkcji liniowej z dodanym szumem losowym z rozkładu normalnego.
# Stwórz instancję klasy LinearRegression, która reprezentuje model regresji liniowej.
# Dopasuj model do danych, używając metody fit(). W tym przypadku użyj tablicy x jako zmiennej niezależnej, a tablicy y jako zmiennej zależnej.
# Wyciągnij współczynnik kierunkowy (slope) z modelu regresji, który jest przechowywany w atrybucie coef_.
# Wyciągnij wyraz wolny (intercept) z modelu regresji, który jest przechowywany w atrybucie intercept_.
# Narysuj punkty na wykresie, używając funkcji scatter() z biblioteki matplotlib.
# Narysuj dopasowaną linię na wykresie, używając funkcji plot() z biblioteki matplotlib. Linia ta ma równanie y = kx + b, gdzie k to współczynnik kierunkowy, a b to wyraz wolny.
# Wyświetl wykres.












