import seaborn as sns
import numpy as np
import pandas as pd

#przygotowanie danych
sns.set_style("darkgrid")
df = pd.read_csv("OTGLF.csv")
# konwersja daty na odpowiedni format
df['Date'] = pd.to_datetime(df['Date'])
# usuwanie wierszy z brakującymi danymi (jeśli są)
df = df.dropna()

#interpolacja volume zapis nowego pliku
df['Volume'] = df['Volume'].replace(0, np.nan)
df['Volume'] = df['Volume'].interpolate(method='linear')
df.to_csv('CDProject.csv', index=False)