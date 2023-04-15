import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits, load_iris, load_wine
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Wczytywanie danych
iris = load_iris()
wine = load_wine()
digits = load_digits()

# Wybierz zbiór danych
#data = iris
data = wine
#data = digits

X = data.data
y = data.target

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Klasyfikatory
classifiers = {
    'k-NN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, clf in classifiers.items():
    # Trenowanie klasyfikatora
    clf.fit(X_train, y_train)

    # Obliczanie prognoz
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Obliczanie metryk
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Zapisywanie wyników
    results[name] = {
        'Confusion Matrix': f'\n{cm}',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC AUC': roc_auc
    }


# Porównanie wyników klasyfikatorów
results_df = pd.DataFrame(results).T


# Przygotowanie danych do wykresów
accuracy_scores = results_df['Accuracy'].tolist()
precision_scores = results_df['Precision'].tolist()
recall_scores = results_df['Recall'].tolist()
f1_scores = results_df['F1'].tolist()
roc_auc_scores = results_df['ROC AUC'].tolist()

# Utworzenie wykresów słupkowych
bar_width = 0.15
index = np.arange(len(classifiers))

fig, ax = plt.subplots()
accuracy_bars = ax.bar(index, accuracy_scores, bar_width, label='Accuracy')
precision_bars = ax.bar(index + bar_width, precision_scores, bar_width, label='Precision')
recall_bars = ax.bar(index + 2 * bar_width, recall_scores, bar_width, label='Recall')
f1_bars = ax.bar(index + 3 * bar_width, f1_scores, bar_width, label='F1')
roc_auc_bars = ax.bar(index + 4 * bar_width, roc_auc_scores, bar_width, label='ROC AUC')

ax.set_xlabel('Klasyfikatory')
ax.set_ylabel('Wartości miar')
ax.set_title('Porównanie klasyfikatorów dla wine')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(classifiers.keys())
ax.legend()

# Dodanie wartości na górze słupków
def add_values_on_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')

add_values_on_bars(accuracy_bars)
add_values_on_bars(precision_bars)
add_values_on_bars(recall_bars)
add_values_on_bars(f1_bars)
add_values_on_bars(roc_auc_bars)

fig.tight_layout()
plt.savefig('Porównanie klasyfikatorów dla wine')
plt.show()

# Zapisanie wyników do pliku CSV
results_df.to_csv('zadanie1_results_wine.csv')

print(tabulate(results_df, headers='keys', tablefmt='pretty')) #czytelniejsze dane

#print(results_df)
