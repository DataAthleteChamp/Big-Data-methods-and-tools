import numpy as np
from scipy.stats import kurtosis
from nolds import hurst_rs, corr_dim, sampen

# Generowanie szeregu czasowego
ts = np.random.normal(loc=0.0, scale=1.0, size=1000)

# Ustawienia dla okien przesuwnych
window_size = 100
step_size = 50 #19 okien przesuwnych
num_windows = int((len(ts) - window_size) / step_size) + 1

# Przechowywanie wyników w listach
means = []
stds = []
maxs = []
mins = []
medians = []
kurtoza = []
entropies = []
fractal_dimensions = []
hurst_exponents = []

# Przetwarzanie szeregu czasowego po oknach przesuwnych
for i in range(num_windows):
    window = ts[i * step_size:i * step_size + window_size]

    # Cechy statystyczne
    means.append(np.mean(window))
    stds.append(np.std(window))
    maxs.append(np.max(window))
    mins.append(np.min(window))
    medians.append(np.median(window))
    kurtoza.append(kurtosis(window))

    # Cechy nieliniowe
    entropies.append(sampen(window))
    fractal_dimensions.append(corr_dim(window, emb_dim=2))
    hurst_exponents.append(hurst_rs(window))

# Wyświetlenie wyników
print("Średnie: ", means)
print("Odchylenia standardowe: ", stds)
print("Wartości maksymalne: ", maxs)
print("Wartości minimalne: ", mins)
print("Mediany: ", medians)
print("Kurtosis: ", kurtoza)
print("Entropie: ", entropies)
print("Wymiar fraktalny: ", fractal_dimensions)
print("Wykładnik Hursta: ", hurst_exponents)




"""

Okno przesuwne to technika stosowana w analizie sygnałów, 
która pozwala na przetwarzanie sygnału w oknach o stałej wielkości, 
które są przesuwane wzdłuż sygnału. Technika ta umożliwia analizę właściwości sygnału 
w różnych fragmentach czasu i pozwala na wykrycie ewentualnych zmian w tych właściwościach.

W przypadku analizy szeregów czasowych okno przesuwne pozwala
 na obliczenie cech statystycznych i nieliniowych na podzbiorach szeregu czasowego 
 o stałej wielkości, co ułatwia porównywanie różnych fragmentów szeregu czasowego 
 i identyfikowanie zmian w tych cechach w czasie.

"""
















# import pandas as pd
# from collections import OrderedDict
# import ipywidgets as widgets
# from IPython.display import display
#
# # Utworzenie listy słowników z wynikami
# results = []
# for i in range(num_windows):
#     results.append(OrderedDict([('mean', means[i]), ('std', stds[i]), ('max', maxs[i]),
#                                 ('min', mins[i]), ('median', medians[i]), ('kurtosis', kurtoses[i]),
#                                 ('entropy', entropies[i]), ('fractal dimension', fractal_dimensions[i]),
#                                 ('hurst exponent', hurst_exponents[i])]))
#
# # Utworzenie ramki danych z wynikami
# df = pd.DataFrame.from_records(results)
#
# # Wyświetlenie ramki danych
# print(df.to_string(index=False))
#



# # Wyświetlenie ramki danych w osobnym oknie
# output = widgets.Output()
# with output:
#     display(df)
# display(output)
#
# # Zapisanie ramki danych do pliku CSV
# df.to_csv('wyniki.csv', index=False)
