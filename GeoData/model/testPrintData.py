#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# import socket as sc

# sc.setdefaulttimeout(10)

dt_weather = pd.read_csv("D:\\UtilityApp\\xampp\\htdocs\\GeoData\\model\\weather_uts.csv", delimiter=',')

test = dt_weather['Location'].unique()
print(test)

# print("PRINT1")
# print("PRINT2")
# print("PRINT3")
# print("PRINT4")
# print("PRINT5")