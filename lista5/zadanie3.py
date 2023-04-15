import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, fetch_california_housing, fetch_openml, load_breast_cancer, \
    load_digits


def reduce_and_plot_lda(data, target, title):
    # Standaryzacja danych
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)

    # Redukcja wymiarów
    lda = LDA(n_components=2)
    reduced_data = lda.fit_transform(data_std, target)

    # Wizualizacja danych
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap='viridis', alpha=0.5)
    plt.xlabel('LDA 1')
    plt.ylabel('LDA 2')
    plt.title(title)
    plt.colorbar()
    plt.savefig(title)
    plt.show()


# Iris dataset
iris = load_iris()
reduce_and_plot_lda(iris.data, iris.target, 'Iris Dataset (LDA)')

# Wine dataset
wine = load_wine()
reduce_and_plot_lda(wine.data, wine.target, 'Wine Dataset (LDA)')

# California Housing dataset
california_housing = fetch_california_housing()
reduce_and_plot_lda(california_housing.data, np.digitize(california_housing.target, np.arange(1, 5)), 'California Housing Dataset (LDA)')

# Red Wine Quality dataset
red_wine_quality = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
numeric_target = red_wine_quality.target.astype(float)
reduce_and_plot_lda(red_wine_quality.data, numeric_target, 'Red Wine Quality Dataset (LDA)')


# # Breast Cancer dataset
# breast_cancer = load_breast_cancer()
# reduce_and_plot_lda(breast_cancer.data, breast_cancer.target, 'Breast Cancer Dataset (LDA)')


# Digits dataset
digits = load_digits()
reduce_and_plot_lda(digits.data, digits.target, 'Digits Dataset (LDA)')