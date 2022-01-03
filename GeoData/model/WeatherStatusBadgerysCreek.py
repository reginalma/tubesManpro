import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status BadgerysCreek
data_badgeryscreek = data_weather[(data_weather['Location'] =="BadgerysCreek")]
data_today_badgeryscreek = data_badgeryscreek[(data_badgeryscreek['Date']=='2017-06-25')]
data_cuaca_badgeryscreek = data_today_badgeryscreek['RainToday']
numpy_data_cuaca_badgeryscreek = data_cuaca_badgeryscreek.to_numpy()
if(numpy_data_cuaca_badgeryscreek == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_badgeryscreek == 'Yes'):
    print('Hujan')