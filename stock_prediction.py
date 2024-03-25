import numpy as np
import matplotlib.pyplot as plt
from stock import get_stock_data
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Input, LSTM
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def prepare_data(ticker_symbol, target_column, types, history_size=20):
    df = get_stock_data(ticker_symbol, 'NO', types)
    df.sort_index(inplace=True)
    df['label'] = df[target_column].shift(-history_size).copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[target_column].values.reshape(-1, 1))

    features = []
    labels = []
    for i in range(len(scaled_data) - history_size):
        features.append(scaled_data[i:i + history_size])
        labels.append(scaled_data[i + history_size])

    features = np.array(features)
    labels = np.array(labels)

    split = int(len(features) * 0.8)
    x_train, y_train = features[:split], labels[:split]
    x_val, y_val = features[split:], labels[split:]

    return x_train, y_train, x_val, y_val, scaler, df


def build_model(input_shape, mod):
    if mod == 'GRU':
        model = Sequential([
            Input(shape=input_shape),
            GRU(units=50, return_sequences=True),
            Dropout(0.2),
            GRU(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    elif mod == 'LSTM':
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=30):
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=64)
    return history

def model_accuracy(model, x_val, y_val):
    # Calculate predictions on validation set
    y_val_pred = model.predict(x_val)

    # Calculate Mean Squared Error (MSE) on validation set
    mse = mean_squared_error(y_val, y_val_pred)
    print("Mean Squared Error on Validation Set:", mse)

    # Calculate R-squared Score on validation set
    r2 = r2_score(y_val, y_val_pred)
    print("R-squared Score on Validation Set:", r2)

    # Return metrics
    return mse, r2


def predict_next_hours(model, last_sequence, scaler, start_date, type, num):
    predictions = []
    current_sequence = last_sequence.copy()
    prediction_dates = []

    # Determine the next valid prediction date
    next_date = start_date
    if type == 'hourly':
        while len(predictions) < num:
            next_date += pd.Timedelta(hours=1)
            # Check if next date is within open hours
            if next_date.weekday() < 5 and 14 <= next_date.hour < 20:
                prediction_dates.append(next_date)
                next_hour_pred = model.predict(np.expand_dims(current_sequence[-1:], axis=0))
                predictions.append(
                    next_hour_pred[0, 0])  # Assuming the output shape is (batch_size, sequence_length, num_features)
                current_sequence = np.append(current_sequence, next_hour_pred, axis=0)
    elif type == 'daily':
        while len(predictions) < num:
            next_date += pd.Timedelta(days=1)
            # Check if next date is a business day
            if next_date.weekday() < 5:
                prediction_dates.append(next_date)
                next_hour_pred = model.predict(np.expand_dims(current_sequence[-1:], axis=0))
                predictions.append(
                    next_hour_pred[0, 0])  # Assuming the output shape is (batch_size, sequence_length, num_features)
                current_sequence = np.append(current_sequence, next_hour_pred, axis=0)

    # Convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, 1)

    # Inverse transform predictions to original scale
    predictions = scaler.inverse_transform(predictions)

    return predictions, prediction_dates


# GRU for stock prediction
def stock_prediction_gru(ticker_symbol, target_column, type, predict_time, accuracy,mod):
    history_size = 20

    x_train, y_train, x_val, y_val, scaler, df = prepare_data(ticker_symbol, target_column, type, history_size)
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = build_model(input_shape,mod)
    history = train_model(model, x_train, y_train, x_val, y_val, epochs=10)  # Reduced to 10 epochs

    last_sequence = x_val[-1]
    start_date = df.index[-1]  # Get the last date in the dataset
    next_hours_predictions, prediction_dates = predict_next_hours(model, last_sequence, scaler, start_date, type, predict_time)

    for i in range(len(next_hours_predictions)):
        if type == 'hourly':
            print(f"At {prediction_dates[i].strftime('%Y-%m-%d %H:%M:%S')} {target_column} is ${next_hours_predictions[i][0]:.2f}")
        elif type == 'daily':
            print(f"On {prediction_dates[i].strftime('%Y-%m-%d')} {target_column} is ${next_hours_predictions[i][0]:.2f}")

    if accuracy == 'yes':
        model_accuracy(model, x_val, y_val)

    return next_hours_predictions



# example
if __name__ == '__main__':
    stock_prediction_gru('AAPL', 'High', 'hourly', 8, 'no','GRU')
    stock_prediction_gru('AAPL', 'High', 'hourly', 8, 'yes','LSTM')
    stock_prediction_gru('AAPL', 'High', 'daily', 5, 'no','GRU')
    stock_prediction_gru('AAPL', 'High', 'daily', 5, 'yes', 'LSTM')

