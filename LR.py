import main

"""
Predictive Modeling
Supervised Machine Learning Model (Linear Regressio)

The model will be only applied from 2016 onwards. 
"""

# 1. Importing the required libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# 2. Search for the index of the first date registered of 2016
date_search = pd.to_datetime('2016-01-08')  # Defining the search target (2016-01-08)
index = main.df.index[main.df['date'] == date_search]  # Finding the index correspondent to the search
idx = index.item()


# 3. Linear Regression Model
# Preparing data for training and testing
x = main.df.loc[:idx, 'Date'].values.reshape(-1, 1)    # Reshaping the data
y = main.df.loc[:idx, 'close'].values.reshape(-1, 1)   # Reshaping the data

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()       # Creation of the LR Model
model.fit(X_train, y_train)      # Training of the LR Model

y_pred = model.predict(X_test)   # Making predictions based on the test set

# Evaluation of the predictions
mse = mean_squared_error(y_test, y_pred)    # Mean Squared Error
r2 = r2_score(y_test, y_pred)               # R2 Score
# print(f'Mean Squared Error (MSE) : {mse}')
# print(f'Coefficient of Determination (R2) : {r2}')


# 4. Visualization
# Transforming X_test points from ordinal to date objects for representation purposes
points = [dt.datetime.fromordinal(z) for x in X_test.T for z in x]

# Scatter plot for data representation and line plot for Model Predictions
plt.figure(figsize=(11, 7))
plt.scatter(points, y_test, label='Actual Price')
plt.plot(points, y_pred, color='red', label='Predicted Price')
plt.title('Linear Regression | Time vs Close Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Stock Price')
plt.legend(loc='upper left')
plt.savefig('LR.png')

# Making a prediction with the model for a specific date (in this case, it predicts the next day)
predict = model.predict((main.df['Date'].iloc[1] + 30).reshape(-1, 1))
print(predict)