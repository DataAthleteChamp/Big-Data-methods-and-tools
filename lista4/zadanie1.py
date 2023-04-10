import numpy as np
from scipy.stats import skew, kurtosis

ts = np.random.normal(size=500)

# Obliczenie wybranych cech statystycznych

mean = np.mean(ts) #sum()/len()
std_dev = np.std(ts) #sqrt (1/n* sum(xi-sr)^2) - pierwiastek z wariancji =odchylenie standardowe
minimum = np.min(ts)
maximum = np.max(ts)
# for num in nums:
#     if num > max_value:
#         max_value = num

kurtoza = kurtosis(ts)
#skewness = skew(ts) #skosnosc
#abs_mean = np.mean(np.abs(ts))
median = np.median(ts)

nums = [1, 3, 5, 2, 4]
nums_sorted = sorted(nums)

# n = len(nums)
# if n % 2 == 0:  # lista ma parzystą liczbę elementów
#     middle_idx = int(n / 2)
#     median = (nums_sorted[middle_idx - 1] + nums_sorted[middle_idx]) / 2
# else:  # lista ma nieparzystą liczbę elementów
#     middle_idx = int(n / 2)
#     median = nums_sorted[middle_idx]
#
# print(median)  # wyświetli 3

print(ts)
print("Średnia: ", mean)
print("Odchylenie standardowe: ", std_dev)
print("Minimum: ", minimum)
print("Maksimum: ", maximum)
print("Kurtoza: ", kurtoza)
#print("Skośność: ", skewness)
#print("Wartość średnia bezwzględna: ", abs_mean)
print("Mediana:", median)



"""
Średnia arytmetyczna: wartość średnia próbki danych.

Odchylenie standardowe: miara rozproszenia próbki danych wokół wartości średniej.

Wartość maksymalna: największa wartość w próbce danych.

Wartość minimalna: najmniejsza wartość w próbce danych.

Mediana: wartość, która dzieli próbkę danych na dwie równe części.

Skośność: miara asymetrii rozkładu wartości próbki danych.

Kurtoza: miara "spiczastości" rozkładu wartości próbki danych.
"""
