import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Status CoffsHarbour
data_coffsHarbour = data_weather[(data_weather['Location'] =="CoffsHarbour")]
data_today_coffsHarbour = data_coffsHarbour[(data_coffsHarbour['Date']=='2017-06-25')]
data_cuaca_coffsHarbour = data_today_coffsHarbour['RainToday']
numpy_data_cuaca_coffsHarbour = data_cuaca_coffsHarbour.to_numpy()
if(numpy_data_cuaca_coffsHarbour == 'No'):
    print('Cerah') 
elif(numpy_data_cuaca_coffsHarbour == 'Yes'):
    print('Hujan')