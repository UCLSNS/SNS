The system diagram of the chatbot:

![未命名文件](https://github.com/UCLSNS/SNS/assets/160248761/638ff464-4624-40b9-8f7c-782cfa57ccf0)

Brief description of each python file:

1. create_page.py:
Creating GUI elements using the Tkinter library in Python

2. keyword_detect.py
Detect keywords related to stock and weather predictions in a given sentence. The keywords are inputs of prediction functions.

3. client_detect.py
Utilizing the Tkinter library in Python, this code creates a client application for weather and stock prediction chatbothttps://github.com/UCLSNS/SNS/blob/main/README.md

5. server.py
This Python script implements a Flask web application for handling user authentication, stock prediction, 
and weather prediction tasks.

4. preprocessing_data.py
Data preprocessing for data extracted from API, particularly handling missing values and outliers, using linear regression. It includes functions to detect outliers using Z-score, handle outliers using linear regression, handle missing values using linear regression, and perform overall data preprocessing for specified columns in a pandas frame.

stock.py
'''This Python script is designed to fetch historical stock data from Yahoo Finance using the yfinance library. It 
also provides functionality to preprocess the retrieved data, handle missing values, and outliers using linear 
regression.'''

weather.py
This script fetches weather data using the Open-Meteo API for a specified city. It provides functions to retrieve both daily and hourly weather data, preprocess it, and save it to CSV files. Additionally, it includes functions to determine whether weather data is available for a given city and to find the latest date with weather data
user.db
user database: contain password and username

weather_lstm.py
This script provides a quick weather prediction check for a specified city using an LSTM model. It fetches historical weather data, preprocesses it, trains an LSTM model, and predicts future weather conditions. The predictions include maximum temperature, minimum temperature, and rain status for a specified number of days ahead
Some limitations for stock prediction:
1. the number of types (daily/hourly) >1 
2. no enter prediction time
3. enter prediction time = 0
4. hourly: prediction time < 48
5: daily: prediction time < 5
6: no find target column
7. for daily: the number of target columns should <3
8  for daily: not achieve target column in different groups (high, low & open, close & volume)

Some limitations for weather prediction:
1. no city enter
2. city not available in weather API
3. no prediction time (enter 0 or bigger than 48)
4. prediction time & type>2
5. cannot check target columns

Parameters:
- sentence (str): The input sentence to analyze.

Defaults: 
- If the sentence doesn't contain any numbers, the default prediction time is set to 1.
- For daily weather detection, if the predicted feature is weather and whether its maximum or minimum is not specified, both maximum and minimum values are returned.
The following situation would return an error warning:
-For prediction time & type: the number should not be bigger than one.
-For daily prediction of stock:
To maintain accuracy, the number of target columns should not be bigger than 2.
Separate high & low, open & close, volume into different groups, if the target columns contain different groups, it will return an error.
  
status code: 
success:
200: login successful
201: register successful
202: stock prediction successful
error:
400: missing either username or password
401: wrong username or password
402: fail to stock prediction
403: fail to weather prediction
500: fail to connect server

Returns:
For stock predictions:
- target_columns (list): Detected target columns for stock predictions.
- prediction_time (int): Detected prediction time.
- prediction_type (str): Detected prediction type (daily or hourly).

For weather predictions:
- prediction_city (list): Detected city names for which weather predictions are available.
- prediction_city_not_available (list): Detected city names for which weather predictions are not available.
- prediction_time (int): Detected prediction time.
- prediction_type (str): Detected prediction type (daily or hourly).
- target_columns (list): Detected target columns for weather predictions.
"""

