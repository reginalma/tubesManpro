import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status Cobar
data_cobar = data_weather[(data_weather['Location'] =="Cobar")]
data_today_cobar = data_cobar[(data_cobar['Date']=='2017-06-25')]
data_cuaca_cobar = data_today_cobar['RainToday']
numpy_data_cuaca_cobar = data_cuaca_cobar.to_numpy()
if(numpy_data_cuaca_cobar == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_cobar == 'Yes'):
    print('Hujan')