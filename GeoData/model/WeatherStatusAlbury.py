import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status Albury
data_albury = data_weather[(data_weather['Location'] =="Albury")]
data_today_albury = data_albury[(data_albury['Date']=='2017-06-25')]
data_cuaca_albury = data_today_albury['RainToday']
numpy_data_cuaca_albury = data_cuaca_albury.to_numpy()
if(numpy_data_cuaca_albury == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_albury == 'Yes'):
    print('Hujan')