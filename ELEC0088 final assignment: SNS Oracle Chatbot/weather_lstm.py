import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from weather import get_weather_data  # Assuming this is a custom module

'''This script provides a quick weather prediction check for a specified city using an LSTM model. It fetches 
historical weather data, preprocesses it, trains an LSTM model, and predicts future weather conditions. The 
predictions include maximum temperature, minimum temperature, and rain status for a specified number of days ahead'''
def weather_prediction_quick_check(city, prediction_time):
    # Fetch weather data for the specified city
    df, end_date = get_weather_data(city, 'no', 'daily')

    # Convert the date column to datetime format and set it as the index
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d')
    df = df.set_index('TIME')

    # Extract relevant features for prediction and scale them using MinMaxScaler
    data = pd.DataFrame({'TEMP_MAX': df['TEMPERATURE_max'], 'TEMP_MIN': df['TEMPERATURE_min'], 'RAIN': df['RAIN']},
                        index=df.index,
                        columns=['TEMP_MAX', 'TEMP_MIN', 'RAIN'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Prepare the dataset for LSTM modeling
    X = []
    Y_temp_max = []
    Y_temp_min = []
    Y_rain = []
    look_back = 60

    for i in range(len(data_scaled) - look_back - 1):
        X.append(data_scaled[i:(i + look_back), :])
        Y_temp_max.append(data_scaled[i + look_back, 0])
        Y_temp_min.append(data_scaled[i + look_back, 1])
        Y_rain.append(data_scaled[i + look_back, 2])

    X = np.array(X)
    Y_temp_max = np.array(Y_temp_max)
    Y_temp_min = np.array(Y_temp_min)
    Y_rain = np.array(Y_rain)

    trainY = np.column_stack((Y_temp_max, Y_temp_min, Y_rain))

    # Split the dataset into training and testing sets
    train_size = int(len(X) * 0.8)
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = trainY[:train_size], trainY[train_size:]

    # Define and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY,
              epochs=10,
              batch_size=64,
              validation_data=(testX, testY),
              verbose=2)

    # Predict future weather values
    future_temp_max = []
    future_temp_min = []
    future_rain = []
    last_batch = data_scaled[-look_back:]
    last_batch = last_batch.reshape((1, look_back, 3))

    for i in range(prediction_time):
        next_temperature_max, next_temperature_min, next_rain = model.predict(last_batch)[0]

        temp_max = scaler.inverse_transform([[next_temperature_max, 0, 0]])[0][0]
        future_temp_max.append(temp_max)
        temp_min = scaler.inverse_transform([[0, next_temperature_min, 0]])[0][1]
        future_temp_min.append(temp_min)
        future_rain.append(next_rain > 1)
        new_batch = np.append(last_batch[:, 1:, :], [[[next_temperature_max, next_temperature_min, next_rain]]],
                              axis=1)
        last_batch = new_batch.reshape((1, look_back, 3))

    # Convert predicted values to human-readable format
    values = []
    dates = []
    start_date = pd.to_datetime(end_date)

    for i in range(len(future_temp_max)):
        values.append([future_temp_max[i], future_temp_min[i], "Rain" if future_rain[i] else "No Rain"])
        dates.append(start_date + pd.Timedelta(days=i))

    return values, dates


if __name__ == '__main__':
    # Example usage
    values, dates = weather_prediction_quick_check('london', 10)
    print(values, dates)
