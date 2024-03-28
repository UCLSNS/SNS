from flask import Flask, request, jsonify
import sqlite3
from stock_prediction import stock_prediction
from weather_prediction import weather_prediction
import numpy as np
from pandas import Timestamp
from weather_lstm import weather_prediction_quick_check

'''This Python script implements a Flask web application for handling user authentication, stock prediction, 
and weather prediction tasks.'''


app = Flask(__name__)

''': 
status code: 
200: login successful
201: register successful
202: stock prediction successful
400: missing either username or password
401: wrong username or password
402: fail to stock prediction
403: fail to
'''

'''Functions related to login part'''


# Function to check user credentials from the database
def check_credentials(username, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM users WHERE username=? AND password=?''', (username, password))
    result = c.fetchone()
    conn.close()
    return result


def login(username, password):
    if not username or not password:
        return {'error': 'Username and password are required.'}, 400

    result = check_credentials(username, password)
    if result:
        return {'message': 'Login successful.', 'page': 'home'}, 200
    else:
        return {'error': 'Invalid username or password.'}, 401


'''Functions related to register part'''


# Function to create a database table for user registration
def create_table():
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    conn.commit()
    conn.close()


# Function to insert user credentials into the database
def insert_user(username, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()

    # Check if username already exists
    c.execute('''SELECT * FROM users WHERE username=?''', (username,))

    # Insert new user
    c.execute('''INSERT INTO users (username, password) VALUES (?, ?)''', (username, password))
    conn.commit()
    conn.close()
    return {'message': 'Registration successful.', 'page': 'login'}, 201


# Function to handle registration
def register(username, password):
    if not username or not password:
        return {'error': 'Username and password are required.'}, 400
    insert_user(username, password)
    return {'message': 'Registration successful.', 'page': 'login'}, 201


'''Functions related to stock prediction'''


# Function to convert data type
def serialize_data(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert NumPy array to a nested list
    elif isinstance(data, list) and all(isinstance(x, Timestamp) for x in data):
        return [str(x) for x in data]  # Convert Timestamp objects to string representation
    else:
        return data  # Return unchanged if not a NumPy array or list of Timestamps


# Function to stock prediction
def stock(data):
    # Read keyword from client
    ticker_symbol = str(data.get('ticker_symbol', ''))
    target_columns = data.get('target_columns', [])  # Ensure target_columns is a list
    prediction_time = int(data.get('prediction_time', ''))

    type = str(data.get('type', ''))

    # Call stock_prediction function to get date and value
    value, date = stock_prediction(ticker_symbol, target_columns, type, prediction_time, 'no', 'GRU')

    # Create a new dictionary to store modified data
    reply = {}

    # Serialize value
    value = [serialize_data(v) for v in value]

    # Initialize an empty list for each column in target_columns
    for column in target_columns:
        reply[column] = []

    # Check the type and format the dates accordingly
    if type == "daily":
        reply['DATE'] = [d.strftime('%m/%d/%Y') for d in date]
    elif type == "hourly":
        reply['DATE'] = date

    # Append values to the corresponding columns in reply dictionary
    for v in value:
        for col, val in zip(target_columns, v):
            if col == 'Volume':  # Check for the 'Volume' column
                reply[col].append(round(val, 0))
            else:
                reply[col].append(round(val, 2))

    if 'DATE' in reply:
        print(reply)  # success
        return reply, 202
    else:
        return {'error': 'Stock prediction error'}, 402  # fail


'''Function related to weather prediction'''


# Function to weather prediction
def weather(data):
    # Get city list
    cities = data.get('cities', [])  # Ensure cities is a list

    response_data = []  # Initialize list to store response dictionaries for each city

    for city in cities:
        target_columns = data.get('target_columns', [])  # Ensure target_columns is a list
        prediction_time = int(data.get('prediction_time', ''))
        type = str(data.get('type', ''))

        if type == 'quick':
            values, dates = weather_prediction_quick_check(city, prediction_time)  # Quick check prediction model

        else:
            # Call weather_prediction function to get values and dates for the city
            values, dates = weather_prediction(city, target_columns, type, prediction_time, 'no', 'GRU')

        reply = {}
        if type == "daily" or type == 'quick':
            reply['DATE'] = [d.strftime('%m/%d/%Y') for d in dates]
        elif type == "hourly":
            reply['DATE'] = [str(d) for d in dates]

        # Initialize an empty list for each column in target_columns
        for column in target_columns:
            reply[column] = []

        # Append values to the corresponding columns in reply dictionary
        for value_list in values:
            for col, val in zip(target_columns, value_list):
                if type != 'quick':
                    if col == 'RAIN':  # Check for the 'RAIN' column
                        if val > 0.5:  # Consider a threshold for rainfall
                            reply[col].append('rain')
                        else:
                            reply[col].append('no rain')
                    else:
                        reply[col].append(round(val, 2))
                elif type == 'quick':
                    if col == 'RAIN':  # Check for the 'RAIN' column
                        reply[col].append(val)
                    else:
                        reply[col].append(round(val, 2))

        # Append the reply for the current city to the response data
        response_data.append(reply)

        # Check if any response data is available
    if response_data:
        print(response_data)
        for entry in response_data:
            for key, value in entry.items():
                if isinstance(value, list):
                    entry[key] = [float(x) if isinstance(x, np.float32) else x for x in value]

        return response_data, 203  # Return the response data with status code 203 (Accepted)
    else:
        return {
            'error': 'Weather prediction error'}, 403  # Return an error response if no data is available


'''Server'''


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    page = data.get('page', '')
    # According to different page info to different function
    if page == 'login':
        username = data.get('username', '')
        password = data.get('password', '')
        return login(username, password)
    elif page == 'register':
        username = data.get('username', '')
        password = data.get('password', '')
        return register(username, password)
    elif page == 'stock':
        return stock(data)
    elif page == 'weather':
        return weather(data)
    else:
        return {'error': 'Invalid request.'}, 400


if __name__ == '__main__':
    create_table()
    app.run(debug=True)
