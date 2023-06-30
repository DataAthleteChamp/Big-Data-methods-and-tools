from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("CDProject.csv")
df['Date'] = pd.to_datetime(df['Date'])

df = df[['Date', 'Close']]
df = df.set_index('Date')

# Split the data into train and test set
train, test = df[0:-100], df[-100:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on training set only.
scaler.fit(train.values.reshape(-1,1))

# Apply transform to both the training set and the test set.
train_scaled = scaler.transform(train.values.reshape(-1,1))
test_scaled = scaler.transform(test.values.reshape(-1,1))

# Prepare dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
x_train, y_train = create_dataset(train_scaled, look_back)
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))

# Create LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Predicting
trainPredict = model.predict(x_train)
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])

# Making predictions for the next 100 days
x_forecast = train_scaled[-100:] # take last 100 from training set
x_forecast = x_forecast.reshape(-1, 1, 1) # reshape for the model
forecastPredict = model.predict(x_forecast) # forecast
forecastPredict = scaler.inverse_transform(forecastPredict) # inverse transform

# Plotting results
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Close'], color='blue', label='Rzeczywiste wartości')
#plt.plot(df.index[:len(trainPredict)], trainPredict, color='green', label='Training Fit')
plt.plot(df.index[len(df)-100:], forecastPredict, color='red', label='Przewidywane wartości')
plt.title('Predykcja model LSTM')
plt.xlabel('lata')
plt.ylabel('Cena zamknięcia')
plt.legend()
plt.savefig('lstm.png')
plt.show()


# y_test from actual data
y_test = np.array(test['Close']).reshape(-1,1)

# Calculating errors
mae = mean_absolute_error(y_test, forecastPredict)
mse = mean_squared_error(y_test, forecastPredict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, forecastPredict)

print(f'mean_absolute_error: {mae}')
print(f'mean_squared_error: {mse}')
print(f'rmse : {rmse}')
print(f'r2_score: {r2}')
