import numpy as np
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, estimate_bandwidth
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score, homogeneity_score, \
    adjusted_mutual_info_score

# Generowanie danych
blobs, _ = make_blobs(n_samples=300, centers=4, random_state=42)
moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Wybierz zbiór danych
data = moons
dataset_choice = 'moons'  # Można zmienić na 'moons'

if dataset_choice == 'blobs':
    data = blobs
    typ = 'blobs'
elif dataset_choice == 'moons':
    data = moons
    typ = 'moons'

# Algorytmy grupowania
#dla blobs
# algorithms = {
#     'K-Means': KMeans(n_clusters=4),
#     'Mean Shift': MeanShift(),
#     'Agglomerative Clustering': AgglomerativeClustering(n_clusters=4)
# }

#dla moons
algorithms = {
    'K-Means': KMeans(n_clusters=4),
    'Mean Shift': MeanShift(bandwidth=estimate_bandwidth(data, quantile=0.2)),
    'Agglomerative Clustering': AgglomerativeClustering(n_clusters=4)
}


results = {}

for name, alg in algorithms.items():
    # Grupowanie
    labels = alg.fit_predict(data)

    # Obliczanie metryk
    silhouette = silhouette_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    rand_index = adjusted_rand_score(_, labels)
    homogeneity = homogeneity_score(_, labels)
    mutual_info = adjusted_mutual_info_score(_, labels)

    # Zapisywanie wyników
    results[name] = {
        'Silhouette': silhouette,
        'Calinski-Harabasz': calinski_harabasz,
        'Rand Index': rand_index,
        'Homogeneity': homogeneity,
        'Mutual Information': mutual_info
    }

    # Przedstawienie graficzne wyników
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', s=50, alpha=0.6, edgecolors='w')
    plt.title(f'{name}')
    plt.colorbar()
    plt.savefig(f"{name}-{typ}")
    plt.show()

# Porównanie wyników algorytmów
results_df = pd.DataFrame(results).T

# Zapisanie wyników do pliku CSV
results_df.to_csv('zadanie2_results_blobs.csv')


# Transpozycja ramki danych, aby metryki były w kolumnach, a algorytmy w indeksie
results_df_transposed = results_df.T.reset_index().melt(id_vars="index", var_name="Algorithm", value_name="Score")

# Utworzenie wykresu skrzypcowego
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 6))

for i, metric in enumerate(results_df.columns):
    metric_data = results_df_transposed[results_df_transposed['index'] == metric]
    sns.violinplot(data=metric_data, x="Algorithm", y="Score", ax=axes[i], inner="stick")
    axes[i].set_title(f'{metric}')

plt.tight_layout()
plt.savefig(f"comparison-{typ}.png")
plt.show()



#print(results_df)
print(tabulate(results_df, headers='keys', tablefmt='pretty'))