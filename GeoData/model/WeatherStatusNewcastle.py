import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status Newcastle
data_newcastle = data_weather[(data_weather['Location'] =="Newcastle")]
data_today_newcastle = data_newcastle[(data_newcastle['Date']=='2017-06-24')]
data_cuaca_newcastle = data_today_newcastle['RainToday']
numpy_data_cuaca_newcastle = data_cuaca_newcastle.to_numpy()
if(numpy_data_cuaca_newcastle == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_newcastle == 'Yes'):
    print('Hujan')