import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


def reduce_and_plot(data, target, title):
    # Standaryzacja danych
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    # Redukcja wymiarów
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_std)

    # Wizualizacja danych
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap='viridis')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(title)
    plt.colorbar()
    plt.savefig(title)
    plt.show()

# Iris dataset
iris_data, iris_target = load_iris(return_X_y=True)
reduce_and_plot(iris_data, iris_target, 'Iris Dataset (PCA)')

# Wine dataset
wine_data, wine_target = load_wine(return_X_y=True)
reduce_and_plot(wine_data, wine_target, 'Wine Dataset (PCA)')

# Breast Cancer dataset
breast_cancer_data, breast_cancer_target = load_breast_cancer(return_X_y=True)
reduce_and_plot(breast_cancer_data, breast_cancer_target, 'Breast Cancer Dataset (PCA)')

# Digits dataset
digits = load_digits()
reduce_and_plot(digits.data, digits.target, 'Digits Dataset (PCA)')