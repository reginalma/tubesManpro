import numpy as np
import pickle
import sys
humidity9am = sys.argv[3]
humidity3am = sys.argv[4]
rainfall = sys.argv[1]
sunshine = sys.argv[2]
"""
humidity9am = 1
humidity3am = 2
rainfall = 3
sunshine = 4
"""

arr = np.array([[rainfall, sunshine, humidity9am, humidity3am]])
pkl_filename1 = "..\\GeoData\\predict\\Model_DT_RainToday.pkl"


# masukkan model raintoday
with open(pkl_filename1, 'rb') as file:
    rain_today_model = pickle.load(file)
predRainToday = rain_today_model.predict(arr)

print(predRainToday[0])