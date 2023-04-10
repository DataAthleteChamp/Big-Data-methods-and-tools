from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Wczytanie danych
wine = load_wine()
X = wine.data

# Standaryzacja danych
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA z dwoma wymiarami
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Wykres
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=wine.target)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
