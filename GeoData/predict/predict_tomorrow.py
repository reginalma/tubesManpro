# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:50:12 2022

@author: lenovo
"""

import numpy as np
import pickle
import sys
humidity9am = sys.argv[3]
humidity3am = sys.argv[4]
rainfall = sys.argv[1]
sunshine = sys.argv[2]

arr = np.array([[rainfall, sunshine, humidity9am, humidity3am]])
pkl_filename2 = "..\\GeoData\\predict\\Model_kNN_RainTomorrow.pkl"

with open(pkl_filename2, 'rb') as file:
    rain_tomorrow_model = pickle.load(file)
    predRainTomorrow = rain_tomorrow_model.predict(arr)

print(predRainTomorrow[0])