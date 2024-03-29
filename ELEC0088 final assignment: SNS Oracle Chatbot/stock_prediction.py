import numpy as np
from stock import get_stock_data
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Input, LSTM
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

'''This Python script is dedicated to building, training, and evaluating deep learning models for stock price 
prediction using historical stock dat'''


# Prepares the data for modeling by fetching stock data, preprocessing it, and splitting it into training, validation,
# and test sets.
def prepare_data(ticker_symbol, target_columns, types, history_size):
    # Got stock dataset
    df = get_stock_data(ticker_symbol, 'NO', types)
    df.sort_index(inplace=True)
    df['label'] = df[target_columns[0]].shift(-history_size).copy()

    # Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[target_columns].values)

    # splitting it into training, validation and test sets
    features = []
    labels = []
    for i in range(len(scaled_data) - history_size):
        features.append(scaled_data[i:i + history_size])
        labels.append(scaled_data[i + history_size])

    features = np.array(features)
    labels = np.array(labels)

    split1 = int(len(features) * 0.6)
    split2 = int(len(features) * 0.8)

    x_train, y_train = features[:split1], labels[:split1]
    x_val, y_val = features[split1:split2], labels[split1:split2]
    x_test, y_test = features[split2:], labels[split2:]

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler, df


#  Builds a deep learning model based on the specified architecture (GRU or LSTM).
def build_model(input_shape, mod, output_dim):
    if mod == 'GRU':
        model = Sequential([
            Input(shape=input_shape),
            GRU(units=50, return_sequences=True),
            Dropout(0.2),
            GRU(units=50),
            Dropout(0.2),
            Dense(output_dim)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    elif mod == 'LSTM':
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(output_dim)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Trains the deep learning model on the training data and validates it on the validation set
def train_model(model, x_train, y_train, x_val, y_val, epochs=20):
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=128)
    return history


# Tests the trained model on the test data and returns the mean squared error and R-squared score.
def test_model(model, x_test, y_test, scaler):
    y_test_pred = model.predict(x_test)
    mse_list = []
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:, i], y_test_pred[:, i])
        mse_list.append(mse)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)
    return mse_list, y_test_pred_inv


#  Predicts multiple days of stock prices using the trained model.
def predict_multiple_days(model, last_sequence, scaler, start_date, type, num_days, output_dim):
    predictions = []
    prediction_dates = []

    for _ in range(num_days):
        if type == 'hourly':
            next_date = start_date + pd.Timedelta(hours=1)
            # Check if next date is within open hours
            while next_date.weekday() >= 5 or next_date.hour < 14 or next_date.hour >= 20:
                next_date += pd.Timedelta(hours=1)
        elif type == 'daily':
            next_date = start_date + pd.Timedelta(days=1)
            # Skip weekends
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)

        prediction_dates.append(next_date)
        next_day_predictions = model.predict(np.expand_dims(last_sequence[-1:], axis=0))
        predictions.append(next_day_predictions[0])

        # Update last sequence for next prediction
        last_sequence = np.append(last_sequence, next_day_predictions, axis=0)
        start_date = next_date

    # Convert predictions to numpy array
    predictions = np.array(predictions)

    # Inverse transform predictions to original scale
    predictions = scaler.inverse_transform(predictions)

    return predictions, prediction_dates


# Main function for stock price prediction. It fetches, preprocesses, trains, and evaluates the model, as well as
# makes predictions for the specified number of days
def stock_prediction(ticker_symbol, target_columns, type, num_days, accuracy, mod):
    if type == 'daily':
        history_size = 10
    elif type == 'hourly':
        history_size = 64
    x_train, y_train, x_val, y_val, x_test, y_test, scaler, df = prepare_data(ticker_symbol, target_columns, type,
                                                                              history_size)
    input_shape = (x_train.shape[1], x_train.shape[2])
    output_dim = len(target_columns)

    model = build_model(input_shape, mod, output_dim)
    history = train_model(model, x_train, y_train, x_val, y_val, epochs=20)

    last_sequence = x_val[-1]
    start_date = df.index[-1]  # Get the last date in the dataset
    next_days_predictions, prediction_dates = predict_multiple_days(model, last_sequence, scaler, start_date, type,
                                                                    num_days, output_dim)

    for i, date in enumerate(prediction_dates):
        print(f"On {date}")
        for j, target_column in enumerate(target_columns):
            print(f"{target_column} is ${next_days_predictions[i][j]:.2f}")

    if accuracy == 'yes':
        mse_val, r2_val, mse_test, r2_test = model_accuracy(model, x_val, y_val, x_test, y_test)
    return next_days_predictions, prediction_dates


def model_accuracy(model, x_val, y_val, x_test, y_test):
    # Calculate predictions on validation set
    y_val_pred = model.predict(x_val)

    # Calculate Mean Squared Error (MSE) on validation set
    mse_val = mean_squared_error(y_val, y_val_pred)
    print("Mean Squared Error on Validation Set:", mse_val)

    # Calculate R-squared Score on validation set
    r2_val = r2_score(y_val, y_val_pred)
    print("R-squared Score on Validation Set:", r2_val)

    # Calculate predictions on test set
    y_test_pred = model.predict(x_test)

    # Calculate Mean Squared Error (MSE) on test set
    mse_test = mean_squared_error(y_test, y_test_pred)
    print("Mean Squared Error on Test Set:", mse_test)

    # Calculate R-squared Score on test set
    r2_test = r2_score(y_test, y_test_pred)
    print("R-squared Score on Test Set:", r2_test)

    return mse_val, r2_val, mse_test, r2_test


# Update stock_prediction_multiple_days function


# Example usage
if __name__ == '__main__':
    stock_prediction('AAPL', ['Open'], 'daily', 5, 'yes', 'LSTM')
    # stock_prediction('AAPL', ['High', 'Low','Volume','Close','Open'], 'hourly', 48, 'yes', 'GRU')
    stock_prediction('AAPL', ['Open'], 'hourly', 5, 'yes', 'LSTM')
    stock_prediction('AAPL', ['Open', 'High'], 'hourly', 5, 'yes', 'GRU')
    stock_prediction('AAPL', ['Open','Close'], 'hourly', 5, 'yes', 'LSTM')