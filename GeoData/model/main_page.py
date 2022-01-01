#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 18:42:31 2021

@author: rgnalma
"""
# PENGAMBILAN DATA UNTUK HALAMAN MAIN PAGE

import numpy as np
import pandas as pd 

#Baca file csv simpan sebagai objek dataframe dt_weather
dt_weather = pd.read_csv('weather_uts.csv', delimiter = ',')

# ===================================================================================================
# Kota Albury

dt_albury = dt_weather[(dt_weather['Location'] =="Albury")]

# Ambil data pada tanggal 25 Juni 2017 (diasumsikan hari ini)
today_albury = dt_albury[(dt_albury['Date']=='2017-06-25')]

# Cuaca di Albury pada tanggal 25 Juni 2017
cuaca_albury = today_albury['RainToday']

# Temperature di Albury pada tanggal 25 Juni 2017
temp_albury = today_albury['Temp']

# ===================================================================================================
# Kota Albury

dt_albury = dt_weather[(dt_weather['Location'] =="Albury")]

# Ambil data pada tanggal 25 Juni 2017 (diasumsikan hari ini)
today_albury = dt_albury[(dt_albury['Date']=='2017-06-25')]

# Cuaca di Albury pada tanggal 25 Juni 2017
cuaca_albury = today_albury['RainToday']

# Temperature di Albury pada tanggal 25 Juni 2017
temp_albury = today_albury['Temp9am']