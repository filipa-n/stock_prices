"""
UC:               Advanced Data Science
Project:          Predictive Analysis of Stock Market Trends
Company:          McDonald's (MCD - Stock Symbol)
Analysis Period:  (1999-2023)

Author:           Filipa Neves
Student No:       22207823
Date:             07-12-2023

"""

# Terminal: pip install -r requirements.txt

# 1. Importing the required libraries
import requests
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns


# 2. Accessing the API (Data is extracted weekly and stored in a variable called 'data')
API_Key = 'MZ12AF1T8T9D6F0B'   # API Key
stock_symbol = 'MCD'           # Stock Symbol for McDonald's (MCD)

URL =  f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&outputsize=full&symbol={stock_symbol}&interval=5min&apikey={API_Key}'

r = requests.get(URL)          # Response
data = r.json()                # The content from the variable 'r' is stored in the 'data' variable as a JSON


# 3. Working on 'data'
df = pd.DataFrame(data['Weekly Time Series'])   # The content in 'data' is converted into a DataFrame [5 rows, xxxx columns]
df = df.transpose()                             # Transpose the DataFrame [xxxx rows, 5 columns]
df = df.reset_index().rename(columns={'index': 'date'})   #Rename index 'date' as a Dataframe column [xxxx rows, 6 columns]
df['date'] = pd.to_datetime(df['date'])         # Convert the date column to datetime format

# Renaming Columns
df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume', }, inplace=True)
# before: date | 1. open | 2. high | 3. low | 4. close | 5. volume
# after:  date | open | high | low | close | volume

df['year'] = df['date'].dt.year                 # Crate 'year' column from the 'date' column
df['Date'] = df['date'].map(dt.datetime.toordinal)   # Crete 'Date' column that converts the dates from the 'date' column into whole numbers

# print(df)


# 4. Cleaning the DataFrame
#print(df.head())
#print(df.info())

# print(df.duplicated().sum())   # Checking for Duplicates

# Changing the type of columns (open, high, low, close and volume) from str to float
df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})

# Handeling outliers from the 'close' column
#q1 = df['close'].quantile(0.25)
#q3 = df['close'].quantile(0.75)
#iqr = q3 - q1
#upper_bound = q3 + 1.5 * iqr
#df = df[df['close'] <= upper_bound]


# 5. EDA (Exploratory Data Analysis)
sta = df[['open', 'high', 'low', 'close', 'volume']].describe().T  # mean, median, min, max, quantiles
# print(sta)


# 6. Visualization
# Plot of the 'close' column over time
plt.figure(figsize=(11, 7))
plt.plot(df['date'], df['close'])
plt.grid()
plt.title('Closing Stock Price Over Time')
plt.xlabel('Date in Years')
plt.ylabel('Closing Stock Price')
plt.savefig('plot.png')

# Boxplot
plt.figure(figsize=(11, 7))
plt.title('Closing Stock Prices by Years')
plt.xlabel('Years')
plt.ylabel('Closing Stock Price')
plt.grid()
sns.boxplot(x='year', y='close', data=df)
plt.savefig('boxplot.png')

#heatmap
plt.figure(figsize=(8, 8))
correlation = df[['open', 'high', 'low', 'close']].corr()
sns.heatmap(correlation, annot=True, cmap='Blues')
plt.title('Correlation Between Stock Prices')
plt.savefig('heatmap.png')

#histogram
plt.figure(figsize=(11, 7))
plt.hist(df['close'], bins=range(45, 306, 5), edgecolor='k')
plt.title('Distribution of Closing Stock Price')
plt.xlabel('Closing Stock Price')
plt.ylabel('Frequency')
plt.savefig('histogram.png')