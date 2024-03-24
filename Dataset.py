
import spacy
import pandas as pd
from stock import get_stock_data
from weather import get_weather_data

nlp = spacy.load('en_core_web_sm')
def find_city(sentence):
    doc = nlp(sentence)
    cities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return cities

def extract_weather(sentence,type):
    cities = find_city(sentence)
    for city in cities:
        weather_data = get_weather_data(city,'No',type)
        return weather_data


def extract_stock(ticker_symbol,type) :
    stock_data=get_stock_data(ticker_symbol, 'yes', type)
    return stock_data
#example:
if __name__ == "__main__":
    sentence= input("Enter the sentence")
    weather = extract_weather(sentence,'daily')
    print(weather.head())