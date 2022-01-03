import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp Albury
data_albury = data_weather[(data_weather['Location'] =="Albury")]
data_today_albury = data_albury[(data_albury['Date']=='2017-06-25')]
data_temp_albury = data_today_albury['Temp9am']
numpy_data_temp_albury = data_temp_albury.to_numpy()
print(numpy_data_temp_albury)