import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Wczytanie danych
df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')

# Przygotowanie danych
df = df.dropna()
df['AmountSpent'] = df['Quantity'] * df['UnitPrice']
X = df.groupby('CustomerID').agg({'AmountSpent': sum, 'Quantity': sum, 'Country': 'first'}).reset_index(drop=True)
X = pd.get_dummies(X, columns=['Country'])
X = X.drop(['Country_United Kingdom'], axis=1).values

# Standaryzacja danych
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA z dwoma wymiarami
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Wykres
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('zadanie1c.png')
plt.show()
