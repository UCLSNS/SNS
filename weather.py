import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from preprocessing_data import data_preprocessing
import pytz


# Function to find the location of the input city:
def get_location(city):
    # Geocoding API endpoint
    geocoding_endpoint = "https://maps.googleapis.com/maps/api/geocode/json"

    # Parameters for the geocoding API request
    params = {
        "address": city,
        "key": "AIzaSyCQw7h0KTQg-PiSVpPuOjwBtQkR85BPwkQ",
    }

    # Make the geocoding API request to get latitude and longitude
    response = requests.get(geocoding_endpoint, params=params)
    data = response.json()
    lat = lon = 0

    # Extract latitude and longitude from the response
    if data['status'] == 'OK':
        lat = data['results'][0]['geometry']['location']['lat']
        lon = data['results'][0]['geometry']['location']['lng']
    return lat, lon

# Determine whether the weather is available in the weather dataset
def is_valid_weather(city):
    lat, lon = get_location(city)
    if lat == 0 and lon == 0:
        print(f"No valid location found for {city}.")
        return False
    else:
        # Call the weather API function to check if weather data is available for the given city
        # You may need to adjust this based on how you fetch weather data in your application
        weather_data = weather_daily(city, '2023-01-25', '2023-02-01')
        if weather_data.empty:
            print(f"No valid weather data found for {city}.")
            return False
        else:
            print(f"Weather data is available for {city}.")
            return True


# Function to find the latest date with weather data - Daily
def find_latest_weather_daily(city):
    end_date = datetime.today().strftime('%Y-%m-%d')
    while True:
        data = weather_daily(city, end_date, end_date)
        if not data.empty:
            break
        end_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    return end_date


# Function to find the latest date with weather data - Hourly
def find_latest_weather_hourly(city):
    current_date = datetime.now()
    end_date = current_date.replace(minute=0, second=0, microsecond=0)
    while True:
        data = weather_hourly(city, end_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if not data.empty:
            break
        end_date -= timedelta(hours=1)
    return end_date


# Function to fetch daily weather data
def weather_daily(city, start_date, end_date):
    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    lat, lon = get_location(city)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max",
                  "wind_direction_10m_dominant", "et0_fao_evapotranspiration"],
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(4).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(5).ValuesAsNumpy()

    # Extract daily dates
    start_date = pd.to_datetime(daily.Time(), unit="s", utc=True).date()
    end_date = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).date()
    daily_dates = pd.date_range(start=start_date, end=end_date, inclusive="left")

    daily_data = {
        "TIME": daily_dates,
        "TEMPERATURE_max": daily_temperature_2m_max,
        "TEMPERATURE_min": daily_temperature_2m_min,
        "PRECIPITATION": daily_precipitation_sum,
        "WIND_SPEED_max": daily_wind_speed_10m_max,
        "WIND_DIRECTION": daily_wind_direction_10m_dominant,
        "et0_fao_evapotranspiration": daily_et0_fao_evapotranspiration
    }

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe


def weather_hourly(city, start_date, end_date):
    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    lat, lon = get_location(city)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", "pressure_msl",
                   "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum"],
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_rain = hourly.Variables(3).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
    hourly_pressure_msl = hourly.Variables(5).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(8).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}
    hourly_data["temperature"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["pressure"] = hourly_pressure_msl
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed"] = hourly_wind_speed_10m
    hourly_data["wind_direction"] = hourly_wind_direction_10m

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe


# Function to save data to CSV
def save_csv_weather(city,df, types):
    if types == 'daily':
        csv_file = f"{city}_daily.csv"
        df.to_csv(csv_file, index=False)
        print(f"Daily weather data saved successfully to {csv_file}")
    elif types == 'hourly':
        csv_file = f"{city}_hourly.csv"
        df.to_csv(csv_file, index=False)
        print(f"Hourly weather data saved successfully to {csv_file}")


# Function to get weather data
def get_weather_data(city, save, type):
    # Determine whether weather data is available for the city
    is_available = is_valid_weather(city)

    if is_available:
        # Get weather data based on the specified type
        if type == 'daily':
            # Get 10 years of daily data
            end_date = find_latest_weather_daily(city)
            start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
            df = weather_daily(city, start_date, end_date)
            # Ensure column names match for preprocessing
            df = data_preprocessing(df, ["TEMPERATURE_max", "TEMPERATURE_min", "WIND_SPEED_max",
                                         "WIND_DIRECTION", "PRECIPITATION", "et0_fao_evapotranspiration"])
        elif type == 'hourly':
            # Get 5 years of hourly data
            end_date = find_latest_weather_hourly(city)
            start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
            df = weather_hourly(city, start_date, end_date)
            # Ensure column names match for preprocessing
            df = data_preprocessing(df, ["temperature", "relative_humidity_2m", "rain", "snowfall",
                                         "pressure", "surface_pressure", "wind_speed", "precipitation",
                                         "wind_direction"])

        if save == 'yes':
            # Save the data to a CSV file
            save_csv_weather(city, df, type)
    else:
        print("Weather data is not available for", city)


# Example usage
if __name__ == "__main__":
    get_weather_data('beijing','yes','hourly')
    get_weather_data('london','yes','daily')

