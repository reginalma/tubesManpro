import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp Moree
data_moree = data_weather[(data_weather['Location'] =="Moree")]
data_today_moree = data_moree[(data_moree['Date']=='2017-06-25')]
data_temp_moree = data_today_moree['Temp9am']
numpy_data_temp_moree = data_temp_moree.to_numpy()
print(numpy_data_temp_moree)