import numpy as np
import pickle
import sys

rainfall = sys.argv[1]
sunshine = sys.argv[2]
humidity9am = sys.argv[3]
humidity3am = sys.argv[4]

arr = np.array([[rainfall, sunshine, humidity9am, humidity3am]])
pkl_filename2 = "..\\GeoData\\predict\\Model_DT_RainToday.pkl"

with open(pkl_filename2, 'rb') as file:
    rain_tomorrow_model = pickle.load(file)
    predRainTomorrow = rain_tomorrow_model.predict(arr)

print(predRainTomorrow[0])