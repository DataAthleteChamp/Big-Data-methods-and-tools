import numpy as np
from scipy.stats import kurtosis
from nolds import hurst_rs, corr_dim, sampen

# Generowanie szeregu czasowego
ts = np.random.normal(loc=0.0, scale=1.0, size=1000)

# Normalizacja szeregu czasowego
norm_ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

# Standaryzacja szeregu czasowego
std_ts = (ts - np.mean(ts)) / np.std(ts)

# Ustawienia dla okien przesuwnych
window_size = 100
step_size = 50
num_windows = int((len(ts) - window_size) / step_size) + 1

# Przechowywanie wyników w listach
means = []
stds = []
maxs = []
mins = []
medians = []
kurtoses = []
entropies = []
fractal_dimensions = []
hurst_exponents = []

# Przetwarzanie szeregu czasowego po oknach przesuwnych
for i in range(num_windows):
    window = ts[i * step_size:i * step_size + window_size]
    norm_window = norm_ts[i * step_size:i * step_size + window_size]
    std_window = std_ts[i * step_size:i * step_size + window_size]

    # Cechy statystyczne dla oryginalnego szeregu
    means.append(np.mean(window))
    stds.append(np.std(window))
    maxs.append(np.max(window))
    mins.append(np.min(window))
    medians.append(np.median(window))
    kurtoses.append(kurtosis(window))

    # Cechy nieliniowe dla oryginalnego szeregu
    entropies.append(sampen(window))
    fractal_dimensions.append(corr_dim(window, emb_dim=2))
    hurst_exponents.append(hurst_rs(window))




    # Cechy statystyczne dla znormalizowanego szeregu
    means.append(np.mean(norm_window))
    stds.append(np.std(norm_window))
    maxs.append(np.max(norm_window))
    mins.append(np.min(norm_window))
    medians.append(np.median(norm_window))
    kurtoses.append(kurtosis(norm_window))

    # Cechy nieliniowe dla znormalizowanego szeregu
    entropies.append(sampen(norm_window))
    fractal_dimensions.append(corr_dim(norm_window, emb_dim=2))
    hurst_exponents.append(hurst_rs(norm_window))





    # Cechy statystyczne dla zstandaryzowanego szeregu
    means.append(np.mean(std_window))
    stds.append(np.std(std_window))
    maxs.append(np.max(std_window))
    mins.append(np.min(std_window))
    medians.append(np.median(std_window))
    kurtoses.append(kurtosis(std_window))

    # Cechy nieliniowe dla zstandaryzowanego szeregu
    entropies.append(sampen(std_window))
    fractal_dimensions.append(corr_dim(std_window, emb_dim=2))
    hurst_exponents.append(hurst_rs(std_window))

# Wyświetlenie wyników
print("Średnie: ", means)
print("Odchylenia standardowe: ", stds)
print("Wartości maksymalne: ", maxs)
print("Wartości minimalne: ", mins)
print("Mediany: ", medians)
print("Kurtosis: ", kurtosis)
print("Entropie: ", entropies)
print("Wymiar fraktalny: ", fractal_dimensions)
print("Wykładnik Hursta: ", hurst_exponents)

