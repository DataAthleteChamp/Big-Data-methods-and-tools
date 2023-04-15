import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, fetch_california_housing, fetch_openml, load_breast_cancer, \
    load_digits


def reduce_and_plot_svd(data, target, title):
    # Standaryzacja danych
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    # Redukcja wymiarów
    svd = TruncatedSVD(n_components=2)
    reduced_data = svd.fit_transform(data_std)

    # Wizualizacja danych
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap='viridis', alpha=0.5)
    plt.xlabel('SVD 1')
    plt.ylabel('SVD 2')
    plt.title(title)
    plt.colorbar()
    plt.savefig(title)
    plt.show()

# Iris dataset
iris = load_iris()
reduce_and_plot_svd(iris.data, iris.target, 'Iris Dataset (SVD)')

# Wine dataset
wine = load_wine()
reduce_and_plot_svd(wine.data, wine.target, 'Wine Dataset (SVD)')

# California Housing dataset
california_housing = fetch_california_housing()
reduce_and_plot_svd(california_housing.data, np.digitize(california_housing.target, np.arange(1, 5)), 'California Housing Dataset (SVD)')

# Red Wine Quality dataset
red_wine_quality = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
numeric_target = red_wine_quality.target.astype(float)
reduce_and_plot_svd(red_wine_quality.data, numeric_target, 'Red Wine Quality Dataset (SVD)')
#

# # Breast Cancer dataset
# breast_cancer = load_breast_cancer()
# reduce_and_plot_svd(breast_cancer.data, breast_cancer.target, 'Breast Cancer Dataset (SVD)')


# Digits dataset
digits = load_digits()
reduce_and_plot_svd(digits.data, digits.target, 'Digits Dataset (SVD)')