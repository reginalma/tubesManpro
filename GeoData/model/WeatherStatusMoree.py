import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status Moree
data_moree = data_weather[(data_weather['Location'] =="Moree")]
data_today_moree = data_moree[(data_moree['Date']=='2017-06-25')]
data_cuaca_moree = data_today_moree['RainToday']
numpy_data_cuaca_moree = data_cuaca_moree.to_numpy()
if(numpy_data_cuaca_moree == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_moree == 'Yes'):
    print('Hujan')