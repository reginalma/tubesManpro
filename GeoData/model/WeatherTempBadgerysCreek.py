import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp BadgerysCreek
data_badgeryscreek = data_weather[(data_weather['Location'] =="BadgerysCreek")]
data_today_badgeryscreek = data_badgeryscreek[(data_badgeryscreek['Date']=='2017-06-25')]
data_temp_badgeryscreek = data_today_badgeryscreek['Temp9am']
numpy_data_temp_badgeryscreek = data_temp_badgeryscreek.to_numpy()
print(numpy_data_temp_badgeryscreek)