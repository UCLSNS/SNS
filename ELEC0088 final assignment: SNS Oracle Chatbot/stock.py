import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from preprocessing_data import data_preprocessing

'''This Python script is designed to fetch historical stock data from Yahoo Finance using the yfinance library. It 
also provides functionality to preprocess the retrieved data, handle missing values, and outliers using linear 
regression.'''

# Defines a dictionary stock_symbols mapping company names to their respective ticker symbols
# (which stock can prediction)
stock_symbols = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "Alphabet Inc. (Google)": "GOOGL",
    "Meta Platforms, Inc. (Facebook)": "FB",
    "Tesla, Inc.": "TSLA",
    "JPMorgan Chase & Co.": "JPM",
    "Johnson & Johnson": "JNJ",
    "Berkshire Hathaway Inc.": "BRK-B",
    "Visa Inc.": "V",
    "NVIDIA Corporation": "NVDA",
    "Walmart Inc.": "WMT",
    "Mastercard Incorporated": "MA",
    "UnitedHealth Group Incorporated": "UNH",
    "Procter & Gamble Company": "PG",
    "PayPal Holdings, Inc.": "PYPL",
    "The Home Depot, Inc.": "HD",
    "The Walt Disney Company": "DIS",
    "Alibaba Group Holding Limited": "BABA",
    "Bank of America Corporation": "BAC",
    "Salesforce.com, Inc.": "CRM",
    "AT&T Inc.": "T",
    "Exxon Mobil Corporation": "XOM",
    "Verizon Communications Inc.": "VZ",
    "Comcast Corporation": "CMCSA",
    "Netflix, Inc.": "NFLX",
    "Pfizer Inc.": "PFE",
    "NIKE, Inc.": "NKE",
    "Merck & Co., Inc.": "MRK",
    "Intel Corporation": "INTC",
    "Cisco Systems, Inc.": "CSCO",
    "Oracle Corporation": "ORCL",
    "Chevron Corporation": "CVX",
    "The Coca-Cola Company": "KO",
    "Wells Fargo & Company": "WFC",
    "PepsiCo, Inc.": "PEP",
    "Advanced Micro Devices, Inc.": "AMD",
    "Abbott Laboratories": "ABT",
    "McDonald's Corporation": "MCD",
    "Costco Wholesale Corporation": "COST",
    "The Boeing Company": "BA",
    "Starbucks Corporation": "SBUX",
    "Honeywell International Inc.": "HON",
    "Advance Auto Parts, Inc.": "AAP",
    "United Parcel Service, Inc.": "UPS",
    "Citigroup Inc.": "C",
    "U.S. Bancorp": "USB",
    "Texas Instruments Incorporated": "TXN",
    "Lowe's Companies, Inc.": "LOW",
    "Micron Technology, Inc.": "MU"
}


# Find the latest date in the database (real-time) - Daily
def find_latest_data_daily(ticker_symbol):
    end_date = datetime.today().strftime('%Y-%m-%d')
    while True:
        # Create a yfinance Ticker object
        ticker = yf.Ticker(ticker_symbol)
        # Get historical data for the specific date
        data = ticker.history(start=(datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d'),
                              end=end_date)
        if not data.empty:
            break
        end_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    return end_date


# Find the latest data in the database (real-time) - Hourly
def find_latest_data_hourly(ticker_symbol):
    current_date = datetime.now()

    # Set the time component to xx:00:00
    end_date = current_date.replace(minute=0, second=0, microsecond=0)
    hourly_data = None

    # To find the latest time when contain yfinance data
    while True:
        # Create a yfinance Ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Get historical data for the specific date
        hourly_data = ticker.history(start=end_date - timedelta(hours=1), end=end_date)

        # Check if data is not empty
        if not hourly_data.empty:
            break

        # If data is empty, decrement end_date by 1 hour and try again
        end_date -= timedelta(hours=1)

    return end_date


# Determine whether data is available in system
def is_valid_ticker(ticker_symbol):
    if ticker_symbol in stock_symbols.values():
        return True
    else:
        return False


# Get 10-year daily dataset for stock_data
def stock_daily_data(ticker_symbol):
    # Find the latest data in the database (real-time) - Daily
    end_date = find_latest_data_daily(ticker_symbol)
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=3650)).strftime('%Y-%m-%d')
    # Get daily data
    daily_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    index = daily_data.index
    # Save DataFrame to CSV file with the index in the first column
    df = pd.DataFrame(daily_data)
    return df


# Get 700 days for hourly stock data
def stock_hourly_data(ticker_symbol):
    # Find the latest data in the database (real-time) - Hourly
    end_date = find_latest_data_hourly(ticker_symbol)

    # The requested range must be within the last 730 days
    start_date_london = end_date - timedelta(days=700)
    start_date = start_date_london.astimezone(pytz.utc).strftime('%Y-%m-%d')

    # Get hourly data
    hourly_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1h')
    # As the index is +4:00, turn to london
    hourly_data.index = pd.to_datetime(hourly_data.index).tz_convert('Europe/London')
    df = pd.DataFrame(hourly_data)
    return df


# Save into csv file
def save_csv(ticker_symbol, df, types):
    if types == 'daily':
        csv_file = f"{ticker_symbol}_daily.csv"
        df.to_csv(csv_file, index_label='Date')
        print(f"Data saved successfully to {csv_file}")
    elif types == 'hourly':
        # Save DataFrame to CSV file with the index in the first column
        csv_file = f"{ticker_symbol}_hourly.csv"
        df.to_csv(csv_file)
        print(f"Data saved successful to {csv_file}")


# Get dataset which is used for stock prediction and with data preprocess (missing + outlier)
def get_stock_data(ticker_symbol, save, type):
    # Determine whether ticker symbol is available in dataset:
    determine = is_valid_ticker(ticker_symbol)
    df = []
    if determine is True:
        if type == 'daily':
            df = stock_daily_data(ticker_symbol)
        elif type == 'hourly':
            df = stock_hourly_data(ticker_symbol)
        # whether save as csv file
        # Deal with missing data and outlier by using linear regression
        df = data_preprocessing(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
        if save == 'yes':
            save_csv(ticker_symbol, df, type)
    else:
        print("Please retry")
    return df


# EXAMPLE
if __name__ == "__main__":
    # get_stock_data('ticker_symbol',save= whether save as csv, type= hourly/daily)
    # if ticker_symbol is not in list, print try again
    get_stock_data('AAPL', 'yes', 'hourly')
    get_stock_data('AAPL', 'yes', 'daily')
