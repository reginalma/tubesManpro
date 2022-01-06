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
    data_wind_kotabaru = data_date_kotabaru['WindDir9am'].to_numpy()
    if(data_wind_kotabaru == "W"):
        print("West")
    elif(data_wind_kotabaru == "NNW"):
        print("North-Northwest")
    elif(data_wind_kotabaru == "ENE"):
        print("East-Northeast")
    elif(data_wind_kotabaru == "SW"):
        print("South-West")
    elif(data_wind_kotabaru == "SSE"):
        print("South-Southeast")
    elif(data_wind_kotabaru == "SE"):
        print("South-East")
    elif(data_wind_kotabaru == "S"):
        print("South")
    elif(data_wind_kotabaru == "S"):
        print("South")
    elif(data_wind_kotabaru == "WSW"):
        print("West-Southwest")
    elif(data_wind_kotabaru == "E"):
        print("East")  
    elif(data_wind_kotabaru == "NW"):
        print("North-West")
    elif(data_wind_kotabaru == "ESE"):
        print("East-Southeast") 
    elif(data_wind_kotabaru == "SSW"):
        print("South-Southwest") 
    elif(data_wind_kotabaru == "NNE"):
        print("North-Northeast") 
    elif(data_wind_kotabaru == "N"):
        print("North")
         
    # print(data_wind_kotabaru)
else:
    # Get Data Weather Status Except Newcastle And Katherine
    data_kotabaru = data_weather[(data_weather['Location'] == kota)]
    data_date_kotabaru = data_kotabaru[(data_kotabaru['Date']==date)]
    data_wind_kotabaru = data_date_kotabaru['WindDir9am'].to_numpy()
    if(data_wind_kotabaru == "W"):
        print("West")
    elif(data_wind_kotabaru == "NNW"):
        print("North-Northwest")
    elif(data_wind_kotabaru == "ENE"):
        print("East-Northeast")
    elif(data_wind_kotabaru == "SW"):
        print("South-West")
    elif(data_wind_kotabaru == "SSE"):
        print("South-Southeast")
    elif(data_wind_kotabaru == "SE"):
        print("South-East")
    elif(data_wind_kotabaru == "S"):
        print("South")
    elif(data_wind_kotabaru == "S"):
        print("South")
    elif(data_wind_kotabaru == "WSW"):
        print("West-Southwest")
    elif(data_wind_kotabaru == "E"):
        print("East")  
    elif(data_wind_kotabaru == "NW"):
        print("North-West")
    elif(data_wind_kotabaru == "ESE"):
        print("East-Southeast") 
    elif(data_wind_kotabaru == "SSW"):
        print("South-Southwest") 
    elif(data_wind_kotabaru == "NNE"):
        print("North-Northeast") 
    elif(data_wind_kotabaru == "N"):
        print("North")  
        
    # print(data_wind_kotabaru)