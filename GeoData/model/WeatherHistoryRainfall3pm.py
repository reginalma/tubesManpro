import numpy as np
import pandas as pd
import pickle
import sys

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

kota = str(sys.argv[1])
date = str(sys.argv[2])

if(kota != "Newcastle" and kota != "Katherine"):
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_date_kotabaru = data_kotabaru[(data_kotabaru['Date']==date)]
    data_wind_kotabaru = data_date_kotabaru['Rainfall'].to_numpy()
    print(data_wind_kotabaru)
else:
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_date_kotabaru = data_kotabaru[(data_kotabaru['Date']==date)]
    data_wind_kotabaru = data_date_kotabaru['Rainfall'].to_numpy()
    print(data_wind_kotabaru)