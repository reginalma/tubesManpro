import numpy as np
import pandas as pd
import sys

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

kota = sys.argv[1]

if(kota != "Newcastle" and kota != "Katherine"):
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_today_kotabaru = data_kotabaru[(data_kotabaru['Date']=='2017-06-25')]
    data_temp_kotabaru = data_today_kotabaru['Temp9am']
    numpy_data_temp_kotabaru = data_temp_kotabaru.to_numpy()
    # not_numpy_data_temp_kotabaru = numpy_data_temp_kotabaru.translate({ord('['): None})
    print(numpy_data_temp_kotabaru)
else:
    # Get Data Weather Status Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_today_kotabaru = data_kotabaru[(data_kotabaru['Date']=='2017-06-24')]
    data_temp_kotabaru = data_today_kotabaru['Temp9am']
    numpy_data_temp_kotabaru = data_temp_kotabaru.to_numpy()
    # not_numpy_data_temp_kotabaru = numpy_data_temp_kotabaru.translate({ord('['): None})
    print(numpy_data_temp_kotabaru)