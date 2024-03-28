import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import requests
from create_page import center_window, create_message, create_entry, create_page, create_option, create_message_small, \
    create_table,center_window
from stock import stock_symbols
from keyword_detect import keyword_detect_stock, keyword_detect_weather

''' "Utilizing the Tkinter library in Python, this code creates a client application for weather and stock prediction 
chatbot

Some limitation for stock prediction:
1. the number of type (daily/hourly) >1 
2. no enter prediction time
3. enter prediction time = 0
4. hourly: prediction time < 48
5: daily : prediction time < 5
6: no find target column
7. for daily: the number of target column should <3
8  for daily: not achieve target column in different group (high,low & open,close & volume)

Some limitation for weather prediction:
1. no city enter
2. city not available in weather API
3. no prediction time (enter 0 or bigger than 48)
4. prediction time & type>2
5. cannot check target columns
'''


# This function is used to destroy all widgets in the window
def clear_window(window):

    for widget in window.winfo_children():
        widget.destroy()


# Function to clean frame
def clear_frame(frame):
    for widget in frame.winfo_children():
        widget.destroy()


# Function to create an big entry field in the frame
def create_entry_big(frame, row, height):
    entry = tk.Text(frame, height=height, font=("Helvetica", 12))
    entry.grid(row=row, column=0, columnspan=2, padx=10, pady=5)

    # Define a function to get the text from the entry widget
    def get_text():
        return entry.get("1.0", tk.END).strip()  # Retrieve text from the beginning to the end

    # Return both the entry widget and the get_text function
    return entry, get_text


# Function to submit dictionary which contain keywords and page information to server and
# receive data from server, page which page send request to server
def submit_entry(input_data, page):
    data = {}
    for key, value in input_data.items():
        if isinstance(value, list):
            # If value is a list, convert it to JSON-compatible format
            data[key] = value
        elif hasattr(value, 'get') and callable(getattr(value, 'get')):
            # If value has a 'get' method, assume it's some kind of object with data
            data[key] = value.get()
        else:
            # Otherwise, assume it's a string or other primitive type
            data[key] = value

    data["page"] = page # Add page information to server

    # Connect to server
    url = 'http://localhost:5000/predict'
    try:
        response = requests.post(url, json=data) # Send infro as dictionary to server
        return response.json(), response.status_code # Get response and status code from server
    except requests.exceptions.RequestException as e:
        # Fail connected to the server
        messagebox.showerror("Error", "Failed to connect to the server.")
        return {'error': f'Failed to connect to the server: {e}'}, 500


# Function to create a login page with a username and password entry fields. It also has buttons for logging in and
# registering a new account. It takes a window parameter.
def create_login(window):
    clear_window(window) # Clean window

    # Content of login page
    frame = create_page(window, "Login", 1000, 500)
    create_message(frame, "SIGN IN", 0)

    # Store entry information into dictionary(later send to server):username and password
    input_data = {}
    input_data["username"] = create_entry(frame, title="User name:", row=2)
    input_data["password"] = create_entry(frame, title="Password:", row=3)

    # Button: login (Successful: move to homepage)
    def handle_login():
        # Send entry information (username + password to server) with page infro: login
        response, status_code = submit_entry(input_data, 'login')
        # Login successful
        if status_code == 200:
            page = response.get('page', '')
            messagebox.showinfo("Login Result", "Login Successfully!.")
            # Turn to home page
            create_homepage(window)
        elif status_code == 500: # Fail to connect to server
            messagebox.showerror("Error", "Failed to connect to the server.")
        else: # 400: 'Username and password are required.' & 401: Invalid username or password.
            messagebox.showerror("Error", f"{response.get('error','')}")
    submit_button = ttk.Button(frame, text="Login", command=handle_login)
    submit_button.grid(row=4, columnspan=2, padx=10, pady=10)

    # Button: Register (move to register page)
    submit_button = ttk.Button(frame, text="Register", command=lambda: create_register(window))
    submit_button.grid(row=5, columnspan=2, padx=10, pady=10)


# Function to create a registration page with fields for username and password. It allows users to register a new account.
def create_register(window):
    clear_window(window) # Clear window

    # Content of register page
    frame = create_page(window, "Register", 1000, 500)
    create_message(frame, "REGISTER", 0)

    ## Store entry information into dictionary(later send to server):username and password
    input_data = {}
    input_data["username"] = create_entry(frame, title="User name:", row=2)
    input_data["password"] = create_entry(frame, title="Password:", row=3)

    # Button: register. Successful move to login page
    def handle_register():
        response, status_code = submit_entry(input_data, 'register')
        if status_code == 201:
            messagebox.showinfo("Registration Result", "Registration successful!")
            create_login(window)
        elif status_code == 500:
            messagebox.showerror("Error", "Failed to connect to the server.")
        else: # status_code: 400 missing password or username
            messagebox.showerror("Error", f"{response.get('error','')}")
    submit_button = ttk.Button(frame, text="Register", command=handle_register)
    submit_button.grid(row=4, columnspan=2, padx=10, pady=10)

    # Button: back. Move to login page
    submit_button = ttk.Button(frame, text="Back", command=lambda: create_login(window))
    submit_button.grid(row=5, columnspan=2, padx=10, pady=10)


# Function to create a homepage with options for selecting between weather prediction and stock prediction.
def create_homepage(window):
    clear_window(window) # Clear window

    # Content of home page
    frame = create_page(window, "Homepage", 1000, 500)
    create_message(frame, "SNS CHATBOT", 0)

    # Selecting between weather prediction and stock prediction
    selected_option = create_option(frame, title="PREDICTION:", row=1, option=["WEATHER", "STOCK"])

    # Button: submit (according to different selection item to different page)
    def handle_button_click(selected_option):
        if selected_option == "WEATHER":
            create_weather(window)
        elif selected_option == "STOCK":
            create_stock(window)

    submit_button = ttk.Button(frame, text="SUBMIT", command=lambda: handle_button_click(selected_option.get()))
    submit_button.grid(row=3, columnspan=2, padx=10, pady=10)

    # Button: logout
    submit_button = ttk.Button(frame, text="LOG OUT", command=lambda: create_login(window))
    submit_button.grid(row=4, columnspan=2, padx=10, pady=10)


''' Stock prediction '''
# Function to create stock prediction page
def create_stock(window):
    clear_window(window) # Clear window

    # Content of stock prediction
    frame = create_page(window, "STOCK PREDICTION", 1000, 500)
    create_message(frame, "STOCK PREDICTION", 0)
    create_message_small(frame, "Hi! How can I help you today?", 1)

    # Create the entry widget and get_text function
    entry_widget, get_text = create_entry_big(frame, 2, 4)

    # Select stock symbol
    option = list(stock_symbols.keys())
    ticker_name = create_option(frame, title="STOCK:", row=8, option=option)

    # Create comformation page
    window2 = tk.Tk()
    window2.title("Confirmation page")
    window2.geometry(f"{0}x{0}")

    # Button: sumbit entry and turn to conformation page to whether keywords detected is correct
    def handle_button_click(window2, ticker_name, get_text_func):
        sentence = get_text_func()  # Retrieve the entered sentence using the get_text function
        ticker_symbol = stock_symbols[ticker_name.get()]  # Retrieve the selected option from the option menu

        print("Entered sentence:", sentence)
        print("Selected option:", ticker_symbol)

        # Keyword detected
        target_columns, prediction_time, type = keyword_detect_stock(sentence)

        '''ERROR:
        1. the number of type (daily/hourly) >1 
        2. no enter prediction time
        3. enter prediction time = 0
        4. hourly: prediction time < 48
        5: daily : prediction time < 5
        6: no find target column
        7. for daily: the number of target column should <3
        8  for daily: not achieve target column in different group (high,low & open,close & volume)'''

        if not type:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect prediction time.")
        elif not prediction_time:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect how long you want to predict.")
        elif prediction_time == 0:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot process 0 prediction times .")
        elif prediction_time > 48 and type == 'hourly':
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot process prediction times exceeding 48 "
                                         "hourly.")
        elif prediction_time > 5 and type == 'daily':
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot process prediction times exceeding 5 days.")
        elif not target_columns:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect what features you want to predict.")
        elif target_columns[0] == 'error1':
            messagebox.showinfo("ERROR", "Unfortunately, In order to maintain accuracy, the number of simultaneous "
                                         "predictions made for daily prediction should not exceed 2.")
        elif target_columns[0] == 'error2':
            messagebox.showinfo("ERROR", "To enhance accuracy, if multiple daily predictions are desired, please "
                                         "group 'close' and 'open' together, while 'high' and 'low', and 'volume' "
                                         "should be considered separately.")
        else:
            # Jump confirmation stock
            create_confirmation_stock(window, frame, window2, ticker_symbol, target_columns, prediction_time, type)
    submit_button = ttk.Button(frame, text="SUBMIT",
                               command=lambda: handle_button_click(window2, ticker_name, get_text))
    submit_button.grid(row=9, columnspan=2, padx=10, pady=10)

    # Button: back to homepage
    def handle_back(window):
        window2.destroy()
        create_homepage(window)

    back_button = ttk.Button(frame, text="BACK", command=lambda: handle_back(window))
    back_button.grid(row=10, columnspan=2, padx=10, pady=10)

    # Button: back to login page
    def handle_logout(window):
        window2.destroy()
        create_login(window)

    logout_button = ttk.Button(frame, text="LOG OUT", command=lambda: handle_logout(window))
    logout_button.grid(row=11, columnspan=2, padx=10, pady=10)


# Function to create stock confirmation page
def create_confirmation_stock(window, frame, window2, ticker_symbol, target_columns, prediction_time, type):
    clear_window(window2) # clear window2
    window2.lift(window)

    # Content: draw table which show the keyword detected
    frame2 = create_page(window2, "Confirmation page", 700, 500)
    create_message(frame2, "Confirming stock prediction information:", 0)
    features = ', '.join(target_columns)
    data_dict = {
        "Ticker symbol": [ticker_symbol],
        f"Prediction time ({type})": [prediction_time],
        "Target features": [features]
    }
    create_table(frame2, len(data_dict['Ticker symbol']), data_dict, 1, width=200)

    # Button: submit (submit keyword and page information to server)
    def handle_button_click():
        # The keyword required to submit to server
        input_data = {}
        input_data = {
            'ticker_symbol': ticker_symbol,
            'target_columns': target_columns,
            'prediction_time': int(prediction_time),
            'type': str(type)
        }

        clear_window(window) # Clear window
        window2.destroy()  # Close the confirmation window after submission

        # Waiting page
        frame = create_page(window, "waiting page", 1000, 500)
        create_message(frame, "Please wait...", 2)

        # Submit keyword to server
        response, status_code = submit_entry(input_data, 'stock')

        if status_code == 202: # success get the result from server and turn to result page
            messagebox.showinfo("Stock prediction result", "Stock prediction successful!")
            create_stock_result(window, response, input_data)
        elif status_code == 402: # fail
            messagebox.showerror("Error", response.get('error', ''))
        else:
            messagebox.showerror("Error", "Failed to connect to the server.")
    submit_button = ttk.Button(frame2, text="SUBMIT", command=handle_button_click)
    submit_button.grid(row=10, columnspan=2, padx=10, pady=10)

    # Button: back (move to stock page)
    def back_stock(window):
        window2.destroy()
        create_stock(window)
    submit_button = ttk.Button(frame2, text="Back", command=lambda: back_stock(window))
    submit_button.grid(row=11, columnspan=2, padx=10, pady=10)


# Function to create stock result page
def create_stock_result(window, response, input_data):
    frame = create_page(window, "STOCK PREDICTION RESULT", 1000, 500)

    # Content
    create_message(frame, "Stock prediction result:", 0)

    # Determine whether prediction is for hours or days
    date_unit = 'hours' if input_data['type'] == 'hourly' else 'days'
    prediction_time = str(input_data['prediction_time'])
    stock_symbol = get_key_from_value(stock_symbols, input_data['ticker_symbol'])
    features = ','.join(input_data['target_columns'])

    # Display information about the prediction
    create_message_small(frame,
                         f"There are {prediction_time} {date_unit} of {features} prediction values for {str(stock_symbol)}.",
                         1)

    # Button for navigition
    buttons_frame = ttk.Frame(frame)
    buttons_frame.grid(row=2, columnspan=2, padx=10, pady=10)

    # Button to confirm and return to stock page
    ok_button = ttk.Button(buttons_frame, text="OK", command=lambda: create_stock(window))
    ok_button.grid(row=0, column=0, padx=5)

    # Button to return to homepage
    home_button = ttk.Button(buttons_frame, text="Home", command=lambda: create_homepage(window))
    home_button.grid(row=0, column=1, padx=5)

    # Button to log out
    logout_button = ttk.Button(buttons_frame, text="Log Out", command=lambda: create_login(window))
    logout_button.grid(row=0, column=2, padx=5)

    # Extract the dictionary containing the stock prediction result from the servers
    dictionary = response  # response is a tuple, and the dictionary is the first element

    # Create a table to display the stock prediction result
    # The number of rows is determined by the length of the 'DATE' key in the dictionary
    num_rows = len(dictionary['DATE'])

    # The number of columns is determined by the number of keys in the dictionary
    num_columns = len(dictionary.keys())

    # Calculate the width of each column
    column_width = int(800 / num_columns)

    # Create the table with the calculated number of rows, dictionary, and column width
    create_table(window, num_rows, dictionary, 5, column_width)


# Function to read key from value
def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Value not found in dictionary


''''weather prediction'''


# Function to weather page
def create_weather(window):
    clear_window(window) # clean window

    # Content
    frame = create_page(window, "WEATHER PREDICTION", 1000, 500)
    create_message(frame, "WEATHER PREDICTION", 0)
    create_message_small(frame, "Hi! How can I help you today?", 1)

    # Create the entry widget and get_text function
    entry_widget, get_text = create_entry_big(frame, 2, 5)

    # Create confirmation page
    window2 = tk.Tk()
    window2.title("Confirmation page")
    window2.geometry(f"{0}x{0}")

    # Button to check keyword from entry
    def handle_button_click(window2, get_text_func):

        sentence = get_text_func()  # Retrieve the entered sentence using the get_text function

        print("Entered sentence:", sentence)

        # Keyword detect
        prediction_city, prediction_city_not_available, prediction_time, prediction_type, target_columns = keyword_detect_weather(
            sentence)

        '''
        Error:
        1. no city enter
        2. city not available in weather API
        3. no prediction time (enter 0 or bigger than 48)
        4. prediction time & type>2
        5. cannot check target columns'''

        if len(prediction_city_not_available) != 0:
            not_city = ', '.join(prediction_city_not_available)
            messagebox.showinfo("ERROR",
                                f"Unfortunately, the system cannot find any weather information in {not_city}.")
        elif len(prediction_city) == 0:
            messagebox.showinfo("ERROR",
                                f"Unfortunately, the system cannot find what city you want to predict for.")
        elif not prediction_time:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect how long you want to predict.")
        elif prediction_time > 48 or prediction_time == 0:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot process prediction times exceeding 48 and 0.")
        elif prediction_type is None:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect prediction time.")
        elif not target_columns:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect what features you want to predict.")
        else:
            create_confirmation_weather(window, frame, window2, prediction_city, target_columns, prediction_time,
                                        prediction_type)
    submit_button = ttk.Button(frame, text="SUBMIT",
                               command=lambda: handle_button_click(window2, get_text))
    submit_button.grid(row=3, columnspan=2, padx=10, pady=10)

    # Quick check: type daily & prediction: highest lowest rain
    create_message_small(frame, "QUICK CHECK: please enter city(ies) name and prediction day(s)",4)
    quick_city_entry = create_entry(frame,'City:',5)
    quick_time_entry = create_entry(frame,'Time:',6)

    # Quick check button: success to confirmation window
    def handle_quick_check(window,quick_city_entry,quick_time_entry):
        quick_city=quick_city_entry.get()
        quick_time=quick_time_entry.get()
        print("Entered sentence:", quick_time,quick_city)

        prediction_city, prediction_city_not_available, prediction_time0, prediction_type0, target_columns0 = keyword_detect_weather(
            quick_city)
        prediction_city1, prediction_city_not_available1, prediction_time, prediction_type1, target_columns1 = keyword_detect_weather(
            str(quick_time))

        # The some error with previous one
        if len(prediction_city_not_available) != 0:
            not_city = ', '.join(prediction_city_not_available)
            messagebox.showinfo("ERROR",
                                f"Unfortunately, the system cannot find any weather information in {not_city}.")
        elif len(prediction_city) == 0:
            messagebox.showinfo("ERROR",
                                f"Unfortunately, the system cannot find what city you want to predict for.")
        elif not prediction_time:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot detect how long you want to predict.")
        elif prediction_time > 48 or prediction_time == 0:
            messagebox.showinfo("ERROR", "Unfortunately, the system cannot process prediction times exceeding 48.")
        else:
            target_columns = ['TEMPERATURE_max', 'TEMPERATURE_min', 'RAIN']
            prediction_type='quick'
            create_confirmation_weather(window, frame, window2, prediction_city, target_columns, prediction_time,
                                        prediction_type)
    back_button = ttk.Button(frame, text="QUICK CHECK", command=lambda: handle_quick_check(window,quick_city_entry,quick_time_entry))
    back_button.grid(row=7, columnspan=2, padx=10, pady=10)

    # back to weather page
    def handle_back(window):
        window2.destroy()
        create_homepage(window)
    back_button = ttk.Button(frame, text="BACK", command=lambda: handle_back(window))
    back_button.grid(row=10, columnspan=2, padx=10, pady=10)

    # LOGOUT button
    def handle_logout(window):
        window2.destroy()
        create_login(window)
    logout_button = ttk.Button(frame, text="LOG OUT", command=lambda: handle_logout(window))
    logout_button.grid(row=11, columnspan=2, padx=10, pady=10)


# Create weather confirmation page
def create_confirmation_weather(window, frame, window2, prediction_city, target_columns, prediction_time, type):
    clear_window(window2)
    window2.lift(window)

    # Content
    frame2 = create_page(window2, "Confirmation page", 700, 500)
    create_message(frame2, "Confirming weather prediction information:", 0)

    # Determine the number of rows needed to display the text
    num_rows = max(len(prediction_city), len(target_columns))

    # Create a dictionary with suitable dimensions for the table
    data_dict = {
        "Prediction cities": prediction_city + [''] * (num_rows - len(prediction_city)),
        f"Prediction time ({type})": [prediction_time] + [''] * (num_rows - 1),
        "Target features": target_columns + [''] * (num_rows - len(target_columns))
    }

    # Adjust the height parameter to accommodate the required number of rows
    create_table(frame2, num_rows, data_dict, 1, width=100)

    # Upload keyword information and page infro: weather
    def handle_button_click():
        # Send keyword infro to server
        input_data = {}
        input_data = {
            'cities': prediction_city,
            'target_columns': target_columns,
            'prediction_time': int(prediction_time),
            'type': str(type)
        }
        clear_window(window)
        window2.destroy()  # Close the confirmation window after submission
        frame = create_page(window, "waiting page", 1000, 500)
        create_message(frame, "Please wait...", 2)
        response, status_code = submit_entry(input_data, 'weather')

        # Success to result page
        if status_code == 203:
            messagebox.showinfo("Weather prediction result", " Weather prediction successful!")
            create_weather_result(window, response, input_data)
        # Error
        elif status_code == 403:
            messagebox.showerror("Error", response.get('error', ''))
        else:
            messagebox.showerror("Error", "Failed to connect to the server.")
    submit_button = ttk.Button(frame2, text="SUBMIT", command=handle_button_click)
    submit_button.grid(row=10, columnspan=2, padx=10, pady=10)

    # Back to weather page
    def back_weather(window):
        window2.destroy()
        create_weather(window)
    submit_button = ttk.Button(frame2, text="Back", command=lambda: back_weather(window))
    submit_button.grid(row=11, columnspan=2, padx=10, pady=10)


# Function to weather result
def create_weather_result(window, response, input_data):
    clear_window(window)

    # Content
    frame = create_page(window, "WEATHER PREDICTION RESULT", 1000, 500)

    # Display a message indicating the weather prediction result
    create_message(frame, "Weather Prediction Result:", 0)

    # Determine whether prediction is for hours or days
    date_unit = 'hours' if input_data['type'] == 'hourly' else 'days'
    prediction_time = str(input_data['prediction_time'])
    cities = ', '.join(input_data['cities'])
    features = ', '.join(input_data['target_columns'])

    # Display information about the prediction
    create_message_small(frame,
                         f"There are {prediction_time} {date_unit} of {features} prediction values for {cities}.",
                         1)

    # Create buttons for navigation
    buttons_frame = ttk.Frame(frame)
    buttons_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    # Button to confirm and return to the weather page
    ok_button = ttk.Button(buttons_frame, text="OK", command=lambda: create_weather(window))
    ok_button.grid(row=0, column=0, padx=5)

    # Button to return to homepage
    home_button = ttk.Button(buttons_frame, text="Home", command=lambda: create_homepage(window))
    home_button.grid(row=0, column=1, padx=5)

    # Button to log out
    logout_button = ttk.Button(buttons_frame, text="Log Out", command=lambda: create_login(window))
    logout_button.grid(row=0, column=2, padx=5)

    # Create a frame for displaying city details
    city_details_frame = ttk.Frame(frame)
    city_details_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    # Display the details of each city as button
    button_style = ttk.Style()
    button_style.configure('Weather.TButton', foreground='navy', background='white', font=('Arial', 12),
                           padding=10)
    row = 4
    column =0

    # Create a button for each city to view its details
    for i, city in enumerate(input_data['cities']):
        city_button = ttk.Button(buttons_frame, text=city, style='Weather.TButton',
                                 command=lambda c=city, index=i: create_city_details(window, input_data,response,
                                                                                     index))

        city_button.grid(row=row, column=i, padx=10, pady=5, sticky='ew')
        if column == 3:
            column=0
            row = row +1
        else:
            column = column + 1


# Function to detail infro of weather
def create_city_details(window, input_data,response,i):
    clear_window(window)

    # Content
    frame = create_page(window, f"{input_data['cities'][i]}", 1000, 500)

    # Display a message indicating the stock prediction result
    create_message(frame, "weather prediction result:", 0)

    # Determine whether prediction is for hours or days
    date_unit = 'hours' if input_data['type'] == 'hourly' else 'days'
    prediction_time = str(input_data['prediction_time'])
    cities = input_data['cities'][i]
    features = ','.join(input_data['target_columns'])

    # Display information about the prediction
    create_message_small(frame,
                         f"There are {prediction_time} {date_unit} of {features} prediction values for {cities}.",
                         1)
    # Create buttons for navigation
    buttons_frame = ttk.Frame(frame)
    buttons_frame.grid(row=2, column=0, pady=10)

    # Button to confirm and return to stock page
    ok_button = ttk.Button(buttons_frame, text="OK", command=lambda: create_weather_result(window,response,input_data))
    ok_button.grid(row=0, column=0, padx=5)

    # Button to return to homepage
    home_button = ttk.Button(buttons_frame, text="Home", command=lambda: create_homepage(window))
    home_button.grid(row=0, column=1, padx=5)

    # Button to log out
    logout_button = ttk.Button(buttons_frame, text="Log Out", command=lambda: create_login(window))
    logout_button.grid(row=0, column=2, padx=5)

    # Extract the dictionary containing the stock prediction result from the response
    dictionary = response[i]  # response is a tuple, and the dictionary is the first element


    # Create a table to display the stock prediction result
    # The number of rows is determined by the length of the 'DATE' key in the dictionary
    num_rows = len(dictionary['DATE'])

    # The number of columns is determined by the number of keys in the dictionary
    num_columns = len(dictionary.keys())

    # Calculate the width of each column
    column_width = int(800 / num_columns)

    # Create the table with the calculated number of rows, dictionary, and column width
    create_table(window, num_rows, dictionary, 3, column_width)


# Function to make element of response into 2 significiant point
def format_numerical_values(data):
    formatted_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            formatted_data[key] = [round(x, 2) if isinstance(x, (int, float)) else x for x in value]
        else:
            formatted_data[key] = value
    return formatted_data



if __name__ == "__main__":

    window = tk.Tk()
    create_homepage(window)
    window.mainloop()

