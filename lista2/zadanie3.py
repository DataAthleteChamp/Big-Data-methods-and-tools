import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def wielomian(x):
    return 10*x**3 - 3*x**2 + 20*x + 1

# Generowanie danych
x_data = np.linspace(-2, 2, 11)
y_data = wielomian(x_data)

# Obliczanie funkcji sklejanej 3. stopnia
sklejka = interpolate.splrep(x_data, y_data, k=3)

# Obliczanie wartości funkcji sklejanej dla punktów x_interp
x_interp = np.linspace(-2, 2, 100)
y_interp = interpolate.splev(x_interp, sklejka)

# Generowanie danych dla funkcji
x_wielomian = np.linspace(-2, 2, 1000)
y_wielomian = wielomian(x_wielomian)

plt.plot(x_data, y_data, 'ro', label='Punkty danych')
plt.plot(x_interp, y_interp, 'b-', label='Interpolacja funkcjami sklejanymi')
plt.plot(x_wielomian, y_wielomian,linestyle='dashed', color='yellow', label='Funkcja')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja funkcjami sklejanymi 3. stopnia')
plt.grid(True)
plt.show()
