import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status NorahHead
data_norahHead = data_weather[(data_weather['Location'] =="NorahHead")]
data_today_norahHead = data_norahHead[(data_norahHead['Date']=='2017-06-25')]
data_cuaca_norahHead = data_today_norahHead['RainToday']
numpy_data_cuaca_norahHead = data_cuaca_norahHead.to_numpy()
if(numpy_data_cuaca_norahHead == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_norahHead == 'Yes'):
    print('Hujan')