import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml

def reduce_and_plot(data, target, title):
    # Standaryzacja danych
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    # Redukcja wymiarów
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_std)

    # Wizualizacja danych
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap='viridis', alpha=0.5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(title)
    plt.colorbar()
    plt.savefig(title)
    plt.show()

# California Housing dataset
california_housing = fetch_california_housing()
reduce_and_plot(california_housing.data, california_housing.target, 'California Housing Dataset (PCA)')

# Red Wine Quality dataset
red_wine_quality = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
numeric_target = red_wine_quality.target.astype(float)
reduce_and_plot(red_wine_quality.data, numeric_target, 'Red Wine Quality Dataset (PCA)')
