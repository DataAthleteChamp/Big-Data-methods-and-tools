import pkgutil

import openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Pobranie zbioru danych
mnist = openml.datasets.get_dataset(554)

# Pobranie cech i usunięcie etykiet
pkgutil.get_data()
X,y = mnist.get_data()


# Standaryzacja danych
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Wykres 2D
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, cmap='jet')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('MNIST dataset after PCA')
plt.colorbar()
plt.show()
