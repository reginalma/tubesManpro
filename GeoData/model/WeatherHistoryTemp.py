import numpy as np
import pandas as pd
import pickle
import sys

data_weather = pd.read_csv("..\\GeoData\\model\\weather_uts.csv", delimiter=',')

kota = sys.argv[1]
date = sys.argv[2]

# kota = "Albury"
# date = "2017-06-25"

if(kota != "Newcastle" and kota != "Katherine"):
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_date_kotabaru = data_kotabaru[(data_kotabaru['Date']==date)]
    data_temp_kotabaru = data_date_kotabaru['Temp9am'].to_numpy()
    print(data_temp_kotabaru)
else:
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_date_kotabaru = data_kotabaru[(data_kotabaru['Date']==date)]
    data_temp_kotabaru = data_date_kotabaru['Temp9am'].to_numpy()
    print(data_temp_kotabaru)