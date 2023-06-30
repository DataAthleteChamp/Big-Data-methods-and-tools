import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import nolds

sns.set_style("darkgrid")
df = pd.read_csv("CDProject.csv")
# convert Date to Datetime and make it the index of the DF
df['Datetime'] = df['Date'].apply(lambda x: datetime.fromisoformat(x))
df2 = df.copy(deep=True)
df2.index = df2['Datetime']
df2.drop(['Date', 'Datetime'], inplace=True, axis=1)
df2.plot(subplots=True, figsize=(10,6))
plt.show()


# calculate statistical features
stat_features = df2.describe()
# calculate median
stat_features.loc['median'] = df2.median()
# calculate kurtosis
stat_features.loc['kurtosis'] = df2.kurtosis()

print(stat_features)

# calculate entropy, fractal dimension and Hurst exponent

# calculate entropy
for col in df2.columns:
    p_data= df2[col].value_counts() / len(df2[col])  # calculates the probabilities
    entropy_val = entropy(p_data)  # input probabilities to get the entropy
    print(f'Entropy of {col}: {entropy_val}')

# calculate fractal dimension and Hurst exponent
for col in df2.columns:
    std_df = StandardScaler().fit_transform(np.array(df2[col]).reshape(-1, 1))  # Standardize data
    std_df = std_df.flatten()
    # Fractal Dimension
    fd = nolds.dfa(std_df)
    print(f'Fractal Dimension of {col}: {fd}')
    # Hurst Exponent
    hurst_exp = nolds.hurst_rs(std_df)
    print(f'Hurst Exponent of {col}: {hurst_exp}')



# Save statistical features
stat_features.to_csv('statistical_features.csv')

# Calculate and save entropy, fractal dimension, and Hurst exponent
entropy_values = []
fractal_dimensions = []
hurst_exponents = []

for col in df2.columns:
    p_data= df2[col].value_counts() / len(df2[col])  # calculates the probabilities
    entropy_val = entropy(p_data)  # input probabilities to get the entropy
    entropy_values.append(entropy_val)

    std_df = StandardScaler().fit_transform(np.array(df2[col]).reshape(-1, 1))  # Standardize data
    std_df = std_df.flatten()

    # Fractal Dimension
    fd = nolds.dfa(std_df)
    fractal_dimensions.append(fd)

    # Hurst Exponent
    hurst_exp = nolds.hurst_rs(std_df)
    hurst_exponents.append(hurst_exp)

entropy_df = pd.DataFrame(entropy_values, index=df2.columns, columns=['Entropy'])
entropy_df.to_csv('entropy.csv')

fractal_df = pd.DataFrame(fractal_dimensions, index=df2.columns, columns=['Fractal Dimension'])
fractal_df.to_csv('fractal_dimension.csv')

hurst_df = pd.DataFrame(hurst_exponents, index=df2.columns, columns=['Hurst Exponent'])
hurst_df.to_csv('hurst_exponent.csv')
