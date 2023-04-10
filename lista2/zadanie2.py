import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def wielomian(x):
    return 10*x**3 - 3*x**2 + 20*x + 1

# Generowanie danych
x_data = np.linspace(-2, 2, 4)  # Zmieniamy liczbę punktów na 4, aby użyć wielomianu 3. stopnia
y_data = wielomian(x_data)

# Obliczanie wielomianu interpolacyjnego Lagrange'a
wielomian_lagrange = interpolate.lagrange(x_data, y_data)

# Obliczanie wartości wielomianu interpolacyjnego dla punktów x_interp
x_interp = np.linspace(-2, 2, 10)
y_interp = wielomian_lagrange(x_interp)

# Generowanie danych dla funkcji
x_wielomian = np.linspace(-2, 2, 1000)
y_wielomian = wielomian(x_wielomian)

plt.plot(x_data, y_data, 'ro', label='Punkty danych')
plt.plot(x_interp, y_interp, 'b-', label='Interpolacja Lagrange\'a')
plt.plot(x_wielomian, y_wielomian, linestyle='dashed', color='pink', label='Funkcja')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja wielomianowa 3. stopnia (Lagrange)')
plt.grid(True)
plt.show()
