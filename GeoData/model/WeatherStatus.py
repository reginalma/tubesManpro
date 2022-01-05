import numpy as np
import pandas as pd
import sys

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

kota = sys.argv[1]

if(kota != "Newcastle" and kota != "Katherine"):
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_today_kotabaru = data_kotabaru[(data_kotabaru['Date']=='2017-06-25')]
    data_cuaca_kotabaru = data_today_kotabaru['RainToday']
    numpy_data_cuaca_kotabaru = data_cuaca_kotabaru.to_numpy()
    if(numpy_data_cuaca_kotabaru == 'No'):
        print('Cerah') 
    elif(numpy_data_cuaca_kotabaru == 'Yes'):
        print('Hujan')
else:
    # Get Data Weather Status Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_today_kotabaru = data_kotabaru[(data_kotabaru['Date']=='2017-06-24')]
    data_cuaca_kotabaru = data_today_kotabaru['RainToday']
    numpy_data_cuaca_kotabaru = data_cuaca_kotabaru.to_numpy()
    if(numpy_data_cuaca_kotabaru == 'No'):
        print('Cerah') 
    elif(numpy_data_cuaca_kotabaru == 'Yes'):
        print('Hujan')