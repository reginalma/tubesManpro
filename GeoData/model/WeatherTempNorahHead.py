import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp NorahHead
data_norahHead = data_weather[(data_weather['Location'] =="NorahHead")]
data_today_norahHead = data_norahHead[(data_norahHead['Date']=='2017-06-25')]
data_temp_norahHead = data_today_norahHead['Temp9am']
numpy_data_temp_norahHead = data_temp_norahHead.to_numpy()
print(numpy_data_temp_norahHead)