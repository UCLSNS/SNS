Introduction：
This system utilizes language recognition to identify keywords and performs forecasts for both stocks and weather by extracting datasets from the yfinance library and Open-Meteo API.For stocks, it can predict various features such as high, low, open, close, and volume for up to 50 stocks over multiple days. In terms of weather forecasting, the system can provide predictions for various weather conditions such as temperature, pressure, wind speed, etc., for multiple cities over multiple days. Additionally, the system offers a quick overview feature to predict the maximum temperature, minimum temperature, and rainfall conditions for multiple cities over multiple days.

The system diagram of the chatbot:

![system diagram of chatbot](https://github.com/UCLSNS/SNS/assets/160248761/638ff464-4624-40b9-8f7c-782cfa57ccf0)


Brief description of each python file:
we use LSTM model and GRU model to do multiple predictions of real-time weather of cities all round the world and 50 types of stock price. The whole system is modular design in Python and all the input and output interaction is implemented through the client.

- Client-side
    create_page.py: Creating GUI elements using the Tkinter library in Python

    keyword_detect.py: Detect keywords related to stock and weather predictions in a given sentence. The keywords are inputs of prediction functions.

    client_detect.py: Utilizing the Tkinter library in Python, this code creates a client application for weather and stock prediction

- Server-side:
    server.py: This Python script implements a Flask web application for handling user authentication, stock prediction, and weather prediction tasks.

    user.db: User database to store password and username

    - Dataset fetching:
      stock.py: This Python script is designed to fetch historical stock data from Yahoo Finance using the yfinance library. It also provides functionality to preprocess the retrieved data, a handle missing values, and outliers usi  linear regression.

      weather.py: This script fetches weather data using the Open-Meteo API for a specified city. It provides functions to retrieve both daily and hourly weather data, preprocess it, and save it to CSV files. Additionally, it includes functions to determine whether weather data is available for a given city and to find the latest date with weather data
    
      preprocessing_data.py: Data preprocessing for data extracted from API, particularly handling missing values and outliers, using linear regression. It includes functions to detect outliers using Z-score, handle outliers using linear regression, handle missing values using linear regression, and perform overall data preprocessing for specified columns in a pandas frame.

    - Neural machine learning model:
      stock_prediction.py: This Python script is dedicated to building, training, and evaluating deep learning models for stock prediction using historical weather date.

      weather_prediction.py: This Python script is dedicated to building, training, and evaluating deep learning models for weather prediction using historical weather date
    
      weather_lstm.py: This script provides a quick weather prediction check for a specified city using an LSTM model. It fetches historical weather data, preprocesses it, trains an LSTM model, and predicts future weather conditions. The predictions include maximum temperature, minimum temperature, and rain status for a specified number of days ahead


Language Recognition for Identifying Keywords:

    Keyword Detection:
        For Stock Predictions:
            target_columns (list): Detected target columns for stock predictions.
            prediction_time (int): Detected prediction time.
            prediction_type (str): Detected prediction type (daily or hourly).
        For Weather Predictions:
            prediction_city (list): Detected city names for which weather predictions are available.
            prediction_city_not_available (list): Detected city names for which weather predictions are not available.
            prediction_time (int): Detected prediction time.
            prediction_type (str): Detected prediction type (daily or hourly).
            target_columns (list): Detected target columns for weather predictions.
        Limitations of Detected Keywords:
            The number of prediction types (daily/hourly) should be no more than 2.
            The number of prediction times should not be 1, and the prediction time should not be 0.
            Failure to enter prediction type, target column, or available city name for weather API.
          
        Default:
            Not detect prediction time, prediction time set to be 1.
            For daily temperature prediction, if only temperature enter, prediction both highest and lowest temperature.

Accuracy Limitations:
            For hourly stock prediction, the prediction time should be less than 48.
            
            For daily stock prediction, the prediction time should be less than 5, the number of target columns should be less than 3, and the target columns should not span different groups (e.g., high, low & open, close & volume).
            
            For weather prediction, the prediction time should be less than 48.
            
System State for Connecting to Server:
    Status Codes:
    
            Success:
                200: Login successful.
            
                201: Registration successful.
                
                202: Stock prediction successful.
                
            Error:
                400: Missing either username or password.
                
                401: Wrong username or password.
                
                402: Failed to perform stock prediction.
                
                403: Failed to perform weather prediction.
                
                500: Failed to connect to the server.
        
    
