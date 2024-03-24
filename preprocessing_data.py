import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


# Function to detect outliers using Z-score for a specified column
def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    return df[(z_scores > threshold) | (z_scores < -threshold)]


# Function to handle outliers using linear regression for a specified column
def handle_outliers_with_regression(df, column):
    # Check if there are any outliers in the column
    outliers_df = detect_outliers_zscore(df, column)
    if outliers_df.empty:
        return df
    else:
        # Drop outliers to train the model
        df_clean = df.dropna()
        # Separate data into features (X) and target variable (y)
        X_train = df_clean.index.values.astype(np.int64).reshape(-1, 1)  # Convert datetime to numerical values
        y_train = df_clean[column]
        # Initialize and fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Predict outlier values using linear regression
        X_outliers = outliers_df.index.values.astype(np.int64).reshape(-1, 1)  # Convert datetime to numerical values
        y_predicted = model.predict(X_outliers)
        # Ensure compatibility by explicitly casting to the dtype of the column
        df.loc[outliers_df.index, column] = y_predicted.astype(df[column].dtype)

        return df


# Function to handle missing values using linear regression for a specified column
def handle_missing_with_regression(df, column):
    # Check if there are missing values in the column
    if df[column].isna().any():
        # Drop rows with missing values to train the model
        df_clean = df.dropna(subset=[column])
        # Separate data into features (X) and target variable (y)
        X_train = df_clean.index.values.reshape(-1, 1)
        y_train = df_clean[column]
        # Initialize and fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Predict missing values using linear regression
        X_missing = df[df[column].isna()].index.values.reshape(-1, 1)
        y_predicted = model.predict(X_missing)
        # Update missing values in the dataframe
        # Ensure the predicted values have the same data type as the column
        df.loc[df[column].isna(), column] = y_predicted.astype(df[column].dtype)
    return df



def data_preprocessing(df, columns):
    for column in columns:
        # Handle missing values using linear regression
        df = handle_missing_with_regression(df, column)
        # Handle outlier values using missing data
        df = handle_outliers_with_regression(df, column)

    return df


# Example usage
if __name__ == "__main__":
    # Load data
    ticker_symbol = 'AAPL'
    df = yf.download(ticker_symbol, start='2021-01-01', end='2022-01-01')
    columns = ['High', 'Low']  # Specify the columns to be processed

    df = data_preprocessing(df, columns)
    # df: data set
    # columns: the list of column you want to process

    print(df.head())
