import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp Moree
data_newcastle = data_weather[(data_weather['Location'] =="Newcastle")]
data_today_newcastle = data_newcastle[(data_newcastle['Date']=='2017-06-24')]
data_temp_newcastle = data_today_newcastle['Temp9am']
numpy_data_temp_newcastle = data_temp_newcastle.to_numpy()
print(numpy_data_temp_newcastle)