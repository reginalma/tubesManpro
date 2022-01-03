import numpy as np
import pandas as pd

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

# Get Data Weather Temp CoffsHarbour
data_coffsHarbour = data_weather[(data_weather['Location'] =="CoffsHarbour")]
data_today_coffsHarbour = data_coffsHarbour[(data_coffsHarbour['Date']=='2017-06-25')]
data_temp_coffsHarbour = data_today_coffsHarbour['Temp9am']
numpy_data_temp_coffsHarbour = data_temp_coffsHarbour.to_numpy()
print(numpy_data_temp_coffsHarbour)