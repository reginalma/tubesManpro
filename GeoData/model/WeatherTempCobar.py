import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp Cobar
data_cobar = data_weather[(data_weather['Location'] =="Cobar")]
data_today_cobar = data_cobar[(data_cobar['Date']=='2017-06-25')]
data_temp_cobar = data_today_cobar['Temp9am']
numpy_data_temp_cobar = data_temp_cobar.to_numpy()
print(numpy_data_temp_cobar)