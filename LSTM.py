import main

"""
Predictive Modeling
Long Short-Term Memory (LSTM) Model is a type of Recurrent Neural Network (RNN)

This model will be applied to the entirety of the 'close' column
"""


# 1. Importing the required libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.dates as mdates


# 2. LSTM Model (Long Short Term Memory)
scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize MinMaxScaler
main.df['close'] = scaler.fit_transform(main.df[['close']])  # Scale the 'close' column

# Function to create sequences from the input data
def create_sequences(data, seq_length):
    sequences = []
    targets = []

    # Loop to create sequences and corresponding targets
    for i in range(len(data) - seq_length):
        seq_data = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq_data)
        targets.append(target)

    return np.array(sequences), np.array(targets)

# Define sequence length and create sequences
sequence_length = 10
X, y = create_sequences(main.df['close'].values.reshape(-1, 1), sequence_length)

# Reshape for LSTM input shape (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting the data into  training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[train_size:], X[:train_size]
y_train, y_test = y[train_size:], y[:train_size]

# Create a Sequential Model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model on the training data set
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Making predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to get the real stock prices
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluating the model
mse = mean_squared_error(y_test_actual, predictions)  # Mean Squared error
print(f'Mean Squared Error: {mse}')


# 3. Visualization
# Line Plot for the actual close stock prices and a line plot for the predicted close stock prices
plt.figure(figsize=(16, 8))
plt.plot(main.df.index[-len(y_test_actual):], y_test_actual, label='Actual Stock Price')
plt.plot(main.df.index[-len(predictions):], predictions, label='Predicted Stock Price')
plt.title('Stock Price Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Customize x-axis labels
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Auto-format the date labels for better readability

plt.savefig('LSTM.png')   # Saving the image