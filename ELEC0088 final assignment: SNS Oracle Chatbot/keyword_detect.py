import spacy
from word2number import w2n
# Import function to check if a city is valid for weather prediction
from weather import is_valid_weather

# Load the English language model
nlp = spacy.load('en_core_web_sm')

"""
Detect keywords related to stock and weather predictions in a given sentence.

Parameters:
- sentence (str): The input sentence to analyze.

Defaults: 
- If the sentence doesn't contain any numbers, the default prediction time is set to 1. 
- For daily weather 
detection: if the predicted feature is weather and whether it's maximum or minimum is not specified, both maximum and 
minimum values are returned.

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


# Function to detect keywords related to stock predictions
def keyword_detect_stock(sentence):
    # Tokenize the input sentence
    doc = nlp(sentence.lower())
    # Define target stock attributes and keywords for daily prediction
    target = {'high': 'High', 'low': 'Low', 'open': 'Open', 'close': 'Close', 'volume': 'Volume'}
    keyword_daily = ['daily', 'day', 'week', 'month', 'year']
    columns = list(target.keys())

    # Initialization
    prediction_time = 1
    prediction_type = None
    target_columns = []
    prediction = []
    type = []

    for token in doc:
        # Extract numerical values from the sentence
        if token.like_num:
            # Convert the numerical word to its numeric value
            if token.text in ['one', 'a', 'an']:  # for one
                prediction.append(1)
            else:
                prediction_num = w2n.word_to_num(token.text)
                prediction.append(prediction_num)

    # Determine prediction time and the number of prediction time should not bigger than 1
    if len(prediction) != 1 and len(prediction) != 0:
        prediction_time = None
    elif len(prediction) == 1:
        prediction_time = prediction[0]

    for token in doc:
        # Find target column list
        word = token.text
        for column in columns:
            if column in word and target[column] not in target_columns:
                target_columns.append(target[column])

        # Find prediction type
        if 'hour' in word:
            type.append('hourly')
        elif any(day in word for day in keyword_daily):
            type.append('daily')
            # Turn to prediction time as day form
            if 'week' in word and prediction_time is not None:
                prediction_time *= 7
            if 'month' in word and prediction_time is not None:
                prediction_time *= 30
            if 'year' in word and prediction_time is not None:
                prediction_time *= 365

    # Determine prediction type and the number of prediction type should not bigger than 1
    if len(type) != 1:
        prediction_type = None
    elif len(type) == 1:
        prediction_type = str(type[0])

    # For daily prediction in order to maintain accuracy, for the number of target columns should not bigger than 2 and
    # Separate high & low, open & close, volume into different group, if the target columns contain different group,
    # it will return error.
    if prediction_type == 'daily' and len(target_columns) > 2:
        target_columns = ['error1', 0]
    elif prediction_type == 'daily' and len(target_columns) == 2:
        if 'High' in target_columns and 'Low' not in target_columns:
            target_columns = ['error2', 0]
        elif 'Open' in target_columns and 'Close' not in target_columns:
            target_columns = ['error2', 0]
        elif 'Volume' in target_columns:
            target_columns = ['error2', 0]

    print("Keyword for stock prediction:")
    print(target_columns, prediction_time, prediction_type)
    return target_columns, prediction_time, prediction_type


# Function to find city info in sentence
def find_city(sentence):
    doc = nlp(sentence)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return cities


# Function to keyword detection for weather prediction
def keyword_detect_weather(sentence):
    # Tokenize the input sentence
    doc = nlp(sentence.lower())

    # Find city in the sentence
    prediction_city = []
    prediction_city_not_available = []
    cities = find_city(doc)

    # Check whether city is available in weather API
    for city in cities:
        if is_valid_weather(city):
            prediction_city.append(city)
        else:
            prediction_city_not_available.append(city)

    # Check type
    types = []
    prediction_type = None

    # Check Prediction time
    prediction_time = 1
    prediction = []
    keyword_daily = ['daily', 'day', 'week', 'month', 'year']

    # Check target_columns
    target_hourly = {'tem': "temperature", 'hum': "relative_humidity_2m", 'preci': "RAIN", 'rain': "RAIN",
                     'press': "surface_pressure", 'wind': "wind_speed"}
    max_keyword = ['max', 'highest', 'peak', 'upper limit', 'extreme', 'top', 'apex', 'pinnacle', 'summit']
    min_keyword = ['min', 'lowest', 'bottom', 'least', 'smallest', 'lower limit', 'base', 'nadir']
    target_daily = {'tem': "temperature", 'wind': "WIND_SPEED_max", 'preci': "RAIN", 'rain': "RAIN",
                    'evap': "evapotranspiration"}
    columns_daily = list(target_daily.keys())
    columns_hourly = list(target_hourly.keys())
    target_columns = []

    # Check prediction time
    for token in doc:
        # Check if the token is a numerical word
        if token.like_num:
            # Convert the numerical word to its numeric value
            if token.text in ['one', 'a', 'an']:  # for one
                prediction.append(1)
            else:
                prediction_num = w2n.word_to_num(token.text)
                prediction.append(prediction_num)

    # The number of prediction time should be <0
    if len(prediction) != 1 and len(prediction) != 0:
        prediction_time = None
    elif len(prediction) == 1:
        prediction_time = prediction[0]

    # Check target columns
    # Reset max and min flags for each toke
    is_max = False
    is_min = False

    for token in doc:
        # Find type
        word = token.text
        if 'hour' in word:
            types.append('hourly')
        elif any(day in word for day in keyword_daily):
            types.append('daily')

            # Check if the token contains a numerical value (days, weeks, etc.)
            if 'week' in word and prediction_time is not None:
                prediction_time *= 7
            if 'month' in word and prediction_time is not None:
                prediction_time *= 30
            if 'year' in word and prediction_time is not None:
                prediction_time *= 365

        # find max or min
        for max_word in max_keyword:
            if max_word in word:
                is_max = True
        for min_word in min_keyword:
            if min_word in word:
                is_min = True

    # The number of type should be smaller than 1
    if len(types) == 1:
        prediction_type = types[0]

    for token in doc:
        word = token.text
        # Find target column list
        if prediction_type == 'hourly':
            for column in columns_hourly:
                # Avoid repeat
                if column in word and target_hourly[column] not in target_columns:
                    target_columns.append(target_hourly[column])
        elif prediction_type == 'daily':
            for column in columns_daily:
                if column in word and target_daily[column] not in target_columns:
                    # Determine the type of temperature(max or min)
                    if target_daily[column] == 'temperature' and is_min and is_max:
                        target_columns.append('TEMPERATURE_max')
                        target_columns.append('TEMPERATURE_min')
                        is_min = False
                        is_max = False
                    elif target_daily[column] == 'temperature' and is_max:
                        target_columns.append('TEMPERATURE_max')
                        is_max = False
                    elif target_daily[column] == 'temperature' and is_min:
                        target_columns.append('TEMPERATURE_min')
                        is_min = False
                    elif target_daily[column] == 'temperature' and is_min == False and is_max == False:
                        target_columns.append('TEMPERATURE_max')
                        target_columns.append('TEMPERATURE_min')
                        is_min = False
                        is_max = False
                    else:
                        target_columns.append(target_daily[column])

    print(prediction_city, prediction_city_not_available, prediction_time, prediction_type, target_columns)
    return prediction_city, prediction_city_not_available, prediction_time, prediction_type, target_columns


# Example usage
if __name__ == '__main__':
    sentence = "open low  daily london shanghai max temperature, min ,prec,hum,press,rain,evap"
    prediction_city, prediction_city_not_available, prediction_time, type, target_columns = keyword_detect_weather(
        sentence)

    print(prediction_city, prediction_city_not_available, prediction_time, type, target_columns)
