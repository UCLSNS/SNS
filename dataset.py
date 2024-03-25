import spacy
import pandas as pd
from stock import get_stock_data
from weather_prediction import weather_prediction
from stock_prediction import stock_prediction

nlp = spacy.load('en_core_web_sm')


def find_city(sentence):
    doc = nlp(sentence)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return cities


def extract_weather(sentence, detect_value,type, predict_time, model):
    cities = find_city(sentence)
    predict_date = []
    prediction_values = []
    for city in cities:
        predict_date = []
        prediction_values = []
        if type == 'daily':
            # detect_value:'TEMPERATURE_max','TEMPERATURE_min'
            # model: 'GRU' 'LSTM'
            # predict_time: random
            predict_date, prediction_values = weather_prediction(city, detect_value, 'daily', predict_time, 'no', model)
        elif type == 'hourly':
            # Hourly: Target column:'temperature','precipitation'
            predict_date, prediction_values = weather_prediction(city, detect_value, 'hourly', predict_time, 'no',
                                                                 model)
    return predict_date, prediction_values


def extract_stock(ticker_symbol, type, predict_time, detect_value, model):
    predict_date = []
    prediction_values = []
    predict_date,prediction_values = stock_prediction(ticker_symbol, detect_value, type, predict_time, 'no', model)
    return predict_date,prediction_values


# example:
if __name__ == "__main__":
    sentence = input("Enter the sentence")
    predict_date, prediction_values = extract_weather(sentence, 'temperature', 'hourly', 5, 'GRU')

