import numpy as np
import matplotlib.pyplot as plt

def wielomian(x):
    return 10*x**3 - 3*x**2 + 20*x + 1

# Generowanie danych
x_data = np.linspace(-2, 2, 11)
y_data = wielomian(x_data)

# Funkcja do przeprowadzenia interpolacji liniowej
# def interpolacja_liniowa(x, x_data, y_data):
#     for i in range(len(x_data) - 1):
#         if x >= x_data[i] and x <= x_data[i+1]:
#             y = y_data[i] + (x - x_data[i]) * (y_data[i+1] - y_data[i]) / (x_data[i+1] - x_data[i])
#             return y
#     raise ValueError("x poza zakresem danych")

# Generowanie danych do interpolacji
# x_interp = np.linspace(-2, 2, 100)
# y_interp = [interpolacja_liniowa(x, x_data, y_data) for x in x_interp]

x_interp = np.linspace(-2, 2, 100)
y_interp = np.interp(x_interp, x_data, y_data)

# Generowanie danych dla funkcji
x_wielomian = np.linspace(-2, 2, 1000)
y_wielomian = wielomian(x_wielomian)

plt.plot(x_data, y_data, 'ro', label='Punkty danych')
plt.plot(x_interp, y_interp, 'b-', label='Interpolacja liniowa')
plt.plot(x_wielomian, y_wielomian, 'g-', label='Funkcja')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolacja liniowa ')
plt.grid(True)
plt.show()
