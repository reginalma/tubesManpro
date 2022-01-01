import numpy as np
import pandas as pd

# import sys
# x = sys.argv[0]
# y = sys.argv[1]
# print(x)
# print(y)

dt_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

print(dt_weather['Location'].unique())

# # Penyiapan Data
# #Menghapus data pada atribut yang memiliki nilai NaN
# dt_weather.dropna(subset = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
#                         'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 
#                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 
#                         'Temp3pm', 'RainToday', 'RainTomorrow'], inplace=True)

# #Menghapus data pada atribut yang memiliki nilai negatif
# dt_weather = dt_weather[(dt_weather.MinTemp >= 0)]

# dt_weather_perlocation = dt_weather[dt_weather['Location']==]
# print(dt_weather_perlocation)

# dt_weather_features = dt_weather[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
#                             'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
#                             'Humidity9am', 'Humidity3pm', 'Pressure9am','Pressure3pm', 
#                             'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']]


# # ===================================================================================================
# # Kota Albury

# dt_albury = dt_weather[(dt_weather['Location'] =="Albury")]

# # Ambil data pada tanggal 25 Juni 2017 (diasumsikan hari ini)
# today_albury = dt_albury[(dt_albury['Date']=='2017-06-25')]

# # Cuaca di Albury pada tanggal 25 Juni 2017
# cuaca_albury = today_albury['RainToday']

# # Temperature di Albury pada tanggal 25 Juni 2017
# temp_albury = today_albury['Temp']
# print(temp_albury)

# ===================================================================================================
# Kota Albury

# dt_albury = dt_weather[(dt_weather['Location'] =="Albury")]

# # Ambil data pada tanggal 25 Juni 2017 (diasumsikan hari ini)
# today_albury = dt_albury[(dt_albury['Date']=='2017-06-25')]

# # Cuaca di Albury pada tanggal 25 Juni 2017
# cuaca_albury = today_albury['RainToday']

# # Temperature di Albury pada tanggal 25 Juni 2017
# temp_albury = today_albury['Temp9am']
# print(temp_albury)