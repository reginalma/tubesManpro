#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:56:57 2021

@author: rgnalma
"""
import numpy as np
import pandas as pd 

#Baca file csv simpan sebagai objek dataframe dt_wine 
dt_weather = pd.read_csv('weather_uts.csv', delimiter = ',')

#Cek df
dt_weather.info()

# PENYIAPAN DATA

# Eksplorasi Data

# Pengecekan nilai min, max, mean pada kolom-kolom numerik 
dt_weather.MinTemp.min() 
dt_weather.MinTemp.max()
dt_weather.MinTemp.mean()

dt_weather.MaxTemp.min()
dt_weather.MaxTemp.max()
dt_weather.MaxTemp.mean()
 
dt_weather.Rainfall.min() 
dt_weather.Rainfall.max()
dt_weather.Rainfall.mean()

dt_weather.Evaporation.min() 
dt_weather.Evaporation.max()
dt_weather.Evaporation.mean()

dt_weather.Sunshine.min() 
dt_weather.Sunshine.max()
dt_weather.Sunshine.mean()

dt_weather.WindGustSpeed.min() 
dt_weather.WindGustSpeed.max()
dt_weather.WindGustSpeed.mean()

dt_weather.WindSpeed9am.min()
dt_weather.WindSpeed9am.max() 
dt_weather.WindSpeed9am.mean()

dt_weather.WindSpeed3pm.min() 
dt_weather.WindSpeed3pm.max()
dt_weather.WindSpeed3pm.mean()

dt_weather.Humidity9am.min()
dt_weather.Humidity9am.max()
dt_weather.Humidity9am.mean()

dt_weather.Humidity3pm.min() 
dt_weather.Humidity3pm.max()
dt_weather.Humidity3pm.mean()

dt_weather.Pressure9am.min() 
dt_weather.Pressure9am.max()
dt_weather.Pressure9am.mean()

dt_weather.Pressure3pm.min() 
dt_weather.Pressure3pm.max()
dt_weather.Pressure3pm.mean()

dt_weather.Cloud9am.min() 
dt_weather.Cloud9am.max()
dt_weather.Cloud9am.mean()

dt_weather.Cloud3pm.min() 
dt_weather.Cloud3pm.max()
dt_weather.Cloud3pm.mean()

dt_weather.Temp9am.min() 
dt_weather.Temp9am.max()
dt_weather.Temp9am.mean()

dt_weather.Temp3pm.min() 
dt_weather.Temp3pm.max()
dt_weather.Temp3pm.mean()

# Eksplorasi dgn visualisasi

import matplotlib.pyplot as plt
import numpy as np

# Distribusi nilai tiap atribut
# Visualisasi dg bloxplot

dt_weather.boxplot(column =['MinTemp'], grid = False)
dt_weather.boxplot(column =['MaxTemp'], grid = False)
dt_weather.boxplot(column =['Rainfall'], grid = False)
dt_weather.boxplot(column =['Evaporation'], grid = False)
dt_weather.boxplot(column =['Sunshine'], grid = False)
dt_weather.boxplot(column =['WindGustSpeed'], grid = False)
dt_weather.boxplot(column =['WindSpeed9am'], grid = False)
dt_weather.boxplot(column =['WindSpeed3pm'], grid = False)
dt_weather.boxplot(column =['Humidity9am'], grid = False)
dt_weather.boxplot(column =['Humidity3pm'], grid = False)
dt_weather.boxplot(column =['Pressure9am'], grid = False)
dt_weather.boxplot(column =['Pressure3pm'], grid = False)
dt_weather.boxplot(column =['Cloud9am'], grid = False)
dt_weather.boxplot(column =['Cloud3pm'], grid = False)
dt_weather.boxplot(column =['Temp9am'], grid = False)
dt_weather.boxplot(column =['Temp3pm'], grid = False)

# Penyiapan Data

#Menghapus data pada atribut yang memiliki nilai NaN
dt_weather.dropna(subset = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 
                        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 
                        'Temp3pm', 'RainToday', 'RainTomorrow'], inplace=True)

#Menghapus data pada atribut yang memiliki nilai negatif
dt_weather = dt_weather[(dt_weather.MinTemp >= 0)]

#cek df
dt_weather.info()

# Import LabelEncoder
from sklearn import preprocessing
# Library untuk feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Library untuk algoritma klasifikasi KNN
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#creating labelEncoder
le = preprocessing.LabelEncoder()

#Buat features dari 16 kolom (kecuali atribut kelas dan atribut non-numerik) pada data weather_uts.csv
dt_weather_features = dt_weather[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                            'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                            'Humidity9am', 'Humidity3pm', 'Pressure9am','Pressure3pm', 
                            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']]

#Ubah label string ke numerik
location_labels = dt_weather[['Location']]
location_label_np = np.array(location_labels)
location_np = location_label_np.ravel()
location_labels_en = le.fit_transform(location_np)

#menambahkan atribut 'Location' ke features
dt_weather_features['Location'] = location_labels_en

windGustDir_labels = dt_weather[['WindGustDir']]
windGustDir_label_np = np.array(windGustDir_labels)
windGustDir_np = windGustDir_label_np.ravel()
windGustDir_labels_en = le.fit_transform(windGustDir_np)

#menambahkan atribut 'WindGustDir' ke features
dt_weather_features['WindGustDir'] = windGustDir_labels_en

windDir9am_labels = dt_weather[['WindDir9am']]
windDir9am_label_np = np.array(windDir9am_labels)
windDir9am_np = windDir9am_label_np.ravel()
windDir9am_labels_en = le.fit_transform(windDir9am_np)

#menambahkan atribut 'WindDir9am' ke features
dt_weather_features['WindDir9am'] = windDir9am_labels_en

windDir3pm_labels = dt_weather[['WindDir3pm']]
windDir3pm_label_np = np.array(windDir3pm_labels)
windDir3pm_np = windDir3pm_label_np.ravel()
windDir3pm_labels_en = le.fit_transform(windDir3pm_np)

#menambahkan atribut 'WindDir3pm' ke features
dt_weather_features['WindDir3pm'] = windDir3pm_labels_en

#features terdiri dari 20 kolom 
#'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
#'Humidity9am', 'Humidity3pm', 'Pressure9am','Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Location',
#'WindGustDir, 'WindDir9am', 'WindDir3pm'

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

# Eksperimen

# Pencarian atribut predictor terbaik
# Atribut kelas: 'RainToday'

#Buat label kelas dari kolom wine_class
weather_labels = dt_weather[['RainToday']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

#Ubah matriks 1 kolom ke 1 baris (supaya dapat menjadi parameter le.fit_transform(.))
label_np= weather_label_np.ravel()

#Ubah label string ke numerik
weather_labels_en = le.fit_transform(label_np)
print(weather_labels_en)

#untuk data yang sudah siap, saatnya dilakukan feature extraction dengan chi-square
X = array_weather_features
Y = weather_labels_en

# ===================================================================================================
# FEATURES SELECTION (2 atribut)

# Feature extraction: Pilih 2 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=2)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (3 atribut)

# Feature extraction: Pilih 3 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=3)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (4 atribut)

# Feature extraction: Pilih 4 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=4)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (5 atribut)

# Feature extraction: Pilih 5 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=5)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (6 atribut)

# Feature extraction: Pilih 6 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=6)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (7 atribut)

# Feature extraction: Pilih 7 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=7)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (8 atribut)

# Feature extraction: Pilih 8 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=8)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (9 atribut)

# Feature extraction: Pilih 9 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=9)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (10 atribut)

# Feature extraction: Pilih 10 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=10)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (11 atribut)

# Feature extraction: Pilih 11 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=11)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (12 atribut)

# Feature extraction: Pilih 12 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=12)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (13 atribut)

# Feature extraction: Pilih 13 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=13)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (14 atribut)

# Feature extraction: Pilih 14 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=14)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (15 atribut)

# Feature extraction: Pilih 15 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=15)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (16 atribut)

# Feature extraction: Pilih 16 atribut prediktor 
selector  = SelectKBest(score_func=chi2, k=16)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (17 atribut)

# Feature extraction: Pilih 17 atribut prediktor 
selector  = SelectKBest(score_func=chi2, k=17)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (18 atribut)

# Feature extraction: Pilih 18 atribut prediktor 
selector  = SelectKBest(score_func=chi2, k=18)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (19 atribut)

# Feature extraction: Pilih 19 atribut prediktor 
selector  = SelectKBest(score_func=chi2, k=19)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# FEATURES SELECTION (20 atribut)

# Feature extraction: Pilih 20 atribut prediktor 
selector  = SelectKBest(score_func=chi2, k=20)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
#Berdasarkan hasil eksperimen di atas, kombinasi atribut prediktor terbaik untuk atribut kelas 'Rain Today' adalah
#2 atribut tersignifikan yaitu atribut 'Rainfall' dan atribut 'Humidity3pm'

#Pemilihan tersebut didasarkan oleh hasil komputasi metriks dari hasil evaluasi seluruh model

#2 atribut tersignifikan di atas memiliki hasil komputasi metriks tertinggi sekaligus nilai akurasi 
#tertinggi di antara hasil komputasi metriks lainnya.

# ===================================================================================================
# Pencarian atribut predictor terbaik
# Atribut kelas: 'RainTomorrow'

#Buat label kelas dari kolom wine_class
weather_labels = dt_weather[['RainTomorrow']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

#Ubah matriks 1 kolom ke 1 baris (supaya dapat menjadi parameter le.fit_transform(.))
label_np= weather_label_np.ravel()

#Ubah label string ke numerik
weather_labels_en = le.fit_transform(label_np)
print(weather_labels_en)

#untuk data yang sudah siap, saatnya dilakukan feature extraction dengan chi-square
X = array_weather_features
Y = weather_labels_en

# ===================================================================================================
# FEATURES SELECTION (2 atribut)

# Feature extraction: Pilih 2 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=2)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (3 atribut)

# Feature extraction: Pilih 3 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=3)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (4 atribut)

# Feature extraction: Pilih 4 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=4)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (5 atribut)

# Feature extraction: Pilih 5 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=5)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (6 atribut)

# Feature extraction: Pilih 6 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=6)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (7 atribut)

# Feature extraction: Pilih 7 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=7)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (8 atribut)

# Feature extraction: Pilih 8 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=8)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (9 atribut)

# Feature extraction: Pilih 9 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=9)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (10 atribut)

# Feature extraction: Pilih 10 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=10)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (11 atribut)

# Feature extraction: Pilih 11 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=11)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (12 atribut)

# Feature extraction: Pilih 12 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=12)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (13 atribut)

# Feature extraction: Pilih 13 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=13)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (14 atribut)

# Feature extraction: Pilih 14 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=14)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (15 atribut)

# Feature extraction: Pilih 15 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=15)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (16 atribut)

# Feature extraction: Pilih 16 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=16)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (17 atribut)

# Feature extraction: Pilih 17 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=17)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (18 atribut)

# Feature extraction: Pilih 18 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=18)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (19 atribut)

# Feature extraction: Pilih 19 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=19)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# FEATURES SELECTION (20 atribut)

# Feature extraction: Pilih 20 atribut prediktor yang paling signifikan
selector  = SelectKBest(score_func=chi2, k=20)
selector.fit(X, Y)

# Ambil kolom yg terpilih (kolom dengan nilai koefisien chi-square terbaik)
cols = selector.get_support(indices=True)
# Buat fitur dataframe dengan kolom paling signifikan
weather_features = dt_weather_features.iloc[:,cols]

# Algoritma NBC 

# Pakai dataframe features di atas untuk membuat model klasifikasi dengan algoritma NBC
array_fitur = np.array(weather_features.values)

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_fitur, weather_labels_en, test_size=0.3)

# Library untuk algoritma klasifikasi Naive Bayes
from sklearn.naive_bayes import GaussianNB

NBC_model_wine = GaussianNB()

#Train the model using the training sets
NBC_model_wine.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = NBC_model_wine.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
#Berdasarkan hasil eksperimen di atas, kombinasi atribut prediktor terbaik untuk atribut kelas 'Rain Tomorrow' adalah
#2 atribut tersignifikan yaitu atribut 'Rainfall' dan atribut 'Humidity3pm'

#Pemilihan tersebut didasarkan oleh hasil komputasi metriks dari hasil evaluasi seluruh model

#2 atribut tersignifikan di atas memiliki hasil komputasi metriks tertinggi sekaligus nilai akurasi 
#tertinggi di antara hasil komputasi metriks lainnya.

# ===================================================================================================
# Algoritma k-NN
# Atribut kelas: 'RainToday'

#Buat features dari kombinasi atribut terbaik yaitu 4 atribut tersignifikan
#'Rainfall', 'Humidity3pm'
dt_weather_features = dt_weather[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']]

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

#Buat label kelas dari kolom 'RainToday'
weather_labels = dt_weather[['RainToday']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

#Ubah matriks 1 kolom ke 1 baris (supaya dapat menjadi parameter le.fit_transform(.))
label_np= weather_label_np.ravel()

#creating labelEncoder
le = preprocessing.LabelEncoder()

#Ubah label string ke numerik
weather_labels_en = le.fit_transform(label_np)
print(weather_labels_en)

# Import train_test_split function
from sklearn.model_selection import train_test_split

X = array_weather_features
Y = weather_labels_en

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_weather_features, weather_labels_en, test_size=0.3) 

#Klasifikasi dataset wine.cvs dgn algoritma k-NN
from sklearn.neighbors import KNeighborsClassifier

# ===================================================================================================
# Algoritma k-NN untuk k = 2

#Buat model dgn jumlah neighbor = k = 2
kNN_model_weather = KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 3

#Buat model dgn jumlah neighbor = k = 3
kNN_model_weather = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 4

#Buat model dgn jumlah neighbor = k = 4
kNN_model_weather = KNeighborsClassifier(n_neighbors=4)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 5

#Buat model dgn jumlah neighbor = k = 5
kNN_model_weather = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 6

#Buat model dgn jumlah neighbor = k = 6
kNN_model_weather = KNeighborsClassifier(n_neighbors=6)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 7

#Buat model dgn jumlah neighbor = k = 7
kNN_model_weather = KNeighborsClassifier(n_neighbors=7)
#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 8

#Buat model dgn jumlah neighbor = k = 8
kNN_model_weather = KNeighborsClassifier(n_neighbors=8)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 9

#Buat model dgn jumlah neighbor = k = 9
kNN_model_weather = KNeighborsClassifier(n_neighbors=9)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 10

#Buat model dgn jumlah neighbor = k = 10
kNN_model_weather = KNeighborsClassifier(n_neighbors=10)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 11

#Buat model dgn jumlah neighbor = k = 11
kNN_model_weather = KNeighborsClassifier(n_neighbors=11)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 12

#Buat model dgn jumlah neighbor = k = 12
kNN_model_weather = KNeighborsClassifier(n_neighbors=12)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 13

#Buat model dgn jumlah neighbor = k = 13
kNN_model_weather = KNeighborsClassifier(n_neighbors=13)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 14

#Buat model dgn jumlah neighbor = k = 14
kNN_model_weather = KNeighborsClassifier(n_neighbors=14)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 15

#Buat model dgn jumlah neighbor = k = 15
kNN_model_weather = KNeighborsClassifier(n_neighbors=15)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 16

#Buat model dgn jumlah neighbor = k = 16
kNN_model_weather = KNeighborsClassifier(n_neighbors=16)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 17

#Buat model dgn jumlah neighbor = k = 17
kNN_model_weather = KNeighborsClassifier(n_neighbors=17)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 18

#Buat model dgn jumlah neighbor = k = 18
kNN_model_weather = KNeighborsClassifier(n_neighbors=18)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 19

#Buat model dgn jumlah neighbor = k = 19
kNN_model_weather = KNeighborsClassifier(n_neighbors=19)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 20

#Buat model dgn jumlah neighbor = k = 20
kNN_model_weather = KNeighborsClassifier(n_neighbors=20)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 21

#Buat model dgn jumlah neighbor = k = 21
kNN_model_weather = KNeighborsClassifier(n_neighbors=21)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 22

#Buat model dgn jumlah neighbor = k = 22
kNN_model_weather = KNeighborsClassifier(n_neighbors=22)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 23

#Buat model dgn jumlah neighbor = k = 23
kNN_model_weather = KNeighborsClassifier(n_neighbors=23)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 24

#Buat model dgn jumlah neighbor = k = 24
kNN_model_weather = KNeighborsClassifier(n_neighbors=24)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 25

#Buat model dgn jumlah neighbor = k = 25
kNN_model_weather = KNeighborsClassifier(n_neighbors=25)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 26

#Buat model dgn jumlah neighbor = k = 26
kNN_model_weather = KNeighborsClassifier(n_neighbors=26)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 27

#Buat model dgn jumlah neighbor = k = 27
kNN_model_weather = KNeighborsClassifier(n_neighbors=27)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 28

#Buat model dgn jumlah neighbor = k = 28
kNN_model_weather = KNeighborsClassifier(n_neighbors=28)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 29

#Buat model dgn jumlah neighbor = k = 29
kNN_model_weather = KNeighborsClassifier(n_neighbors=29)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN untuk k = 30

#Buat model dgn jumlah neighbor = k = 30
kNN_model_weather = KNeighborsClassifier(n_neighbors=30)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

# ===================================================================================================
# Algoritma k-NN
# Atribut kelas: 'RainTomorrow'

#Buat features dari kombinasi atribut terbaik yaitu  4 atribut tersignifikan
#'Rainfall', 'Sunshine', 'Humidity9am', 'Humidity3pm'
dt_weather_features = dt_weather[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']]

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

#Buat label kelas dari kolom 'RainTomorrow'
weather_labels = dt_weather[['RainTomorrow']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

#Ubah matriks 1 kolom ke 1 baris (supaya dapat menjadi parameter le.fit_transform(.))
label_np= weather_label_np.ravel()

#creating labelEncoder
le = preprocessing.LabelEncoder()

#Ubah label string ke numerik
weather_labels_en = le.fit_transform(label_np)
print(weather_labels_en)

# Import train_test_split function
from sklearn.model_selection import train_test_split

X = array_weather_features
Y = weather_labels_en

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_weather_features, weather_labels_en, test_size=0.3) 

#Klasifikasi dataset wine.cvs dgn algoritma k-NN
from sklearn.neighbors import KNeighborsClassifier

# ===================================================================================================
# Algoritma k-NN untuk k = 2

#Buat model dgn jumlah neighbor = k = 2
kNN_model_weather = KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 3

#Buat model dgn jumlah neighbor = k = 3
kNN_model_weather = KNeighborsClassifier(n_neighbors=3)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 4

#Buat model dgn jumlah neighbor = k = 4
kNN_model_weather = KNeighborsClassifier(n_neighbors=4)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 5

#Buat model dgn jumlah neighbor = k = 5
kNN_model_weather = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 6

#Buat model dgn jumlah neighbor = k = 6
kNN_model_weather = KNeighborsClassifier(n_neighbors=6)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 7

#Buat model dgn jumlah neighbor = k = 7
kNN_model_weather = KNeighborsClassifier(n_neighbors=7)
#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 8

#Buat model dgn jumlah neighbor = k = 8
kNN_model_weather = KNeighborsClassifier(n_neighbors=8)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 9

#Buat model dgn jumlah neighbor = k = 9
kNN_model_weather = KNeighborsClassifier(n_neighbors=9)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 10

#Buat model dgn jumlah neighbor = k = 10
kNN_model_weather = KNeighborsClassifier(n_neighbors=10)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 11

#Buat model dgn jumlah neighbor = k = 11
kNN_model_weather = KNeighborsClassifier(n_neighbors=11)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 12

#Buat model dgn jumlah neighbor = k = 12
kNN_model_weather = KNeighborsClassifier(n_neighbors=12)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 13

#Buat model dgn jumlah neighbor = k = 13
kNN_model_weather = KNeighborsClassifier(n_neighbors=13)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 14

#Buat model dgn jumlah neighbor = k = 14
kNN_model_weather = KNeighborsClassifier(n_neighbors=14)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 15

#Buat model dgn jumlah neighbor = k = 15
kNN_model_weather = KNeighborsClassifier(n_neighbors=15)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 16

#Buat model dgn jumlah neighbor = k = 16
kNN_model_weather = KNeighborsClassifier(n_neighbors=16)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 17

#Buat model dgn jumlah neighbor = k = 17
kNN_model_weather = KNeighborsClassifier(n_neighbors=17)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 18

#Buat model dgn jumlah neighbor = k = 18
kNN_model_weather = KNeighborsClassifier(n_neighbors=18)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 19

#Buat model dgn jumlah neighbor = k = 19
kNN_model_weather = KNeighborsClassifier(n_neighbors=19)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 20

#Buat model dgn jumlah neighbor = k = 20
kNN_model_weather = KNeighborsClassifier(n_neighbors=20)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 21

#Buat model dgn jumlah neighbor = k = 21
kNN_model_weather = KNeighborsClassifier(n_neighbors=21)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 22

#Buat model dgn jumlah neighbor = k = 22
kNN_model_weather = KNeighborsClassifier(n_neighbors=22)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 23

#Buat model dgn jumlah neighbor = k = 23
kNN_model_weather = KNeighborsClassifier(n_neighbors=23)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 24

#Buat model dgn jumlah neighbor = k = 24
kNN_model_weather = KNeighborsClassifier(n_neighbors=24)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 25

#Buat model dgn jumlah neighbor = k = 25
kNN_model_weather = KNeighborsClassifier(n_neighbors=25)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 26

#Buat model dgn jumlah neighbor = k = 26
kNN_model_weather = KNeighborsClassifier(n_neighbors=26)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 27

#Buat model dgn jumlah neighbor = k = 27
kNN_model_weather = KNeighborsClassifier(n_neighbors=27)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 28

#Buat model dgn jumlah neighbor = k = 28
kNN_model_weather = KNeighborsClassifier(n_neighbors=28)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 29

#Buat model dgn jumlah neighbor = k = 29
kNN_model_weather = KNeighborsClassifier(n_neighbors=29)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
# Algoritma k-NN untuk k = 30

#Buat model dgn jumlah neighbor = k = 30
kNN_model_weather = KNeighborsClassifier(n_neighbors=30)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

# ===================================================================================================
#nilai k terbaik yang dapat digunakan untuk model k-NN adalah nilai k = 14 karena 
#nilai k = 14 memiliki nilai precision dan akurasi tertinggi dibandingkan dengan yang lainnya yaitu 83%

# ===================================================================================================
# Algoritma Decision Tree
# Atribut kelas: 'RainToday'

import pickle
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import the classification library
from sklearn import tree

#Visualize the Decision Tree model 
#Untuk menambah library python-graphviz pada Anaconda:
#Buka  Anaconda Prompt (as Administrator), lalu jalankan:
# conda install -c anaconda pydotplus

from sklearn.tree import export_graphviz
import pydotplus

#Read wine dataset
# dt_iris = pd.read_csv('wine.csv', delimiter = ',')

#Buat features dari kombinasi atribut terbaik yaitu 6 atribut tersignifikan
#'malic_acid', 'alcalinity_of_ash', 'magnesium', 'flavanoids','color_intensity', 'proline
dt_weather_features = dt_weather[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']]

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

#Buat label kelas dari kolom 'RainToday'
weather_labels = dt_weather[['RainToday']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

X = array_weather_features
Y = weather_label_np

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3) 

# Create/initiate the Decision Tree classifer model
DT_model_weather = tree.DecisionTreeClassifier(criterion='entropy')
# Train Decision Tree Classifer using the 70% of the dataset
DT_model_weather.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = DT_model_weather.predict(X_test)

# Evaluate model using test (30%) dataset, print the accuracy
weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

#======jika sudah memperoleh kriteria model yg akurasinya tinggi,
#buat model finalnya dengan menggunakan seluruh data input yg dimiliki======
DT_model_weather_final = tree.DecisionTreeClassifier(criterion='entropy')
DT_model_weather_final.fit(X,Y)

#Visualize the Decision Tree model 

from six import StringIO
from IPython.display import Image

int_class_names=DT_model_weather_final.classes_
str_class_names = int_class_names.astype(str)

#Nama2 fitur/prediktor di DT diambil dari nama atribut prediktor yang dipakai membuat model
#Begitu juga dengan kelas targetnya
dot_data = export_graphviz(DT_model_weather_final,feature_names=dt_weather_features.columns, class_names=str_class_names, filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

#Create and save the graph of tree as image (PNG format)
graph.write_png("Dtree_weather_model.png")
Image(graph.create_png())

# ===================================================================================================
# Algoritma Decision Tree
# Atribut kelas: 'RainTomorrow'

import pickle
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import the classification library
from sklearn import tree

#Visualize the Decision Tree model 
#Untuk menambah library python-graphviz pada Anaconda:
#Buka  Anaconda Prompt (as Administrator), lalu jalankan:
# conda install -c anaconda pydotplus

from sklearn.tree import export_graphviz
import pydotplus

#Read wine dataset
# dt_iris = pd.read_csv('wine.csv', delimiter = ',')

#Buat features dari kombinasi atribut terbaik yaitu  4 atribut tersignifikan
#'Rainfall', 'Sunshine', 'Humidity9am', 'Humidity3pm'
dt_weather_features = dt_weather[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']]

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

#Buat label kelas dari kolom 'RainToday'
weather_labels = dt_weather[['RainTomorrow']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

X = array_weather_features
Y = weather_label_np

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3) 

# Create/initiate the Decision Tree classifer model
DT_model_weather = tree.DecisionTreeClassifier(criterion='entropy')
# Train Decision Tree Classifer using the 70% of the dataset
DT_model_weather.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = DT_model_weather.predict(X_test)

# Evaluate model using test (30%) dataset, print the accuracy
weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

#======jika sudah memperoleh kriteria model yg akurasinya tinggi,
#buat model finalnya dengan menggunakan seluruh data input yg dimiliki======
DT_model_weather_final = tree.DecisionTreeClassifier(criterion='entropy')
DT_model_weather_final.fit(X,Y)

#Visualize the Decision Tree model 

from six import StringIO
from IPython.display import Image

int_class_names=DT_model_weather_final.classes_
str_class_names = int_class_names.astype(str)

#Nama2 fitur/prediktor di DT diambil dari nama atribut prediktor yang dipakai membuat model
#Begitu juga dengan kelas targetnya
dot_data = export_graphviz(DT_model_weather_final,feature_names=dt_weather_features.columns, class_names=str_class_names, filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

#Create and save the graph of tree as image (PNG format)
graph.write_png("Dtree_weather_model.png")
Image(graph.create_png())

# ===================================================================================================
# MODEL TERBAIK

# Atribut kelas: 'RainToday'
# Algoritma Decision Tree

import pickle
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import the classification library
from sklearn import tree

#Visualize the Decision Tree model 
#Untuk menambah library python-graphviz pada Anaconda:
#Buka  Anaconda Prompt (as Administrator), lalu jalankan:
# conda install -c anaconda pydotplus

from sklearn.tree import export_graphviz
import pydotplus

#Read wine dataset
# dt_iris = pd.read_csv('wine.csv', delimiter = ',')

#Buat features dari kombinasi atribut terbaik yaitu  4 atribut tersignifikan
#'Rainfall', 'Sunshine', 'Humidity9am', 'Humidity3pm'
dt_weather_features = dt_weather[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']]

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

#Buat label kelas dari kolom 'RainToday'
weather_labels = dt_weather[['RainToday']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

X = array_weather_features
Y = weather_label_np

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3) 

# Create/initiate the Decision Tree classifer model
DT_model_weather = tree.DecisionTreeClassifier(criterion='entropy')
# Train Decision Tree Classifer using the 70% of the dataset
DT_model_weather.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = DT_model_weather.predict(X_test)

# Evaluate model using test (30%) dataset, print the accuracy
weather_classes = weather_labels.RainToday.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainToday'))

#======jika sudah memperoleh kriteria model yg akurasinya tinggi,
#buat model finalnya dengan menggunakan seluruh data input yg dimiliki======
DT_model_weather_final = tree.DecisionTreeClassifier(criterion='entropy')
DT_model_weather_final.fit(X,Y)

#Visualize the Decision Tree model 

from six import StringIO
from IPython.display import Image

int_class_names=DT_model_weather_final.classes_
str_class_names = int_class_names.astype(str)

#Nama2 fitur/prediktor di DT diambil dari nama atribut prediktor yang dipakai membuat model
#Begitu juga dengan kelas targetnya
dot_data = export_graphviz(DT_model_weather_final,feature_names=dt_weather_features.columns, class_names=str_class_names, filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

#Create and save the graph of tree as image (PNG format)
graph.write_png("Dtree_weather_model.png")
Image(graph.create_png())

#====== SIMPAN MODEL ========

#Simpan model dgn nama: Model_DT_RainToday.pkl
pkl_filename = "Model_DT_RainToday.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(DT_model_weather_final, file)

#============== pemanfaatan model ================
    
# Cara:
#Load/baca model yg sudah disimpan
import pickle
pkl_filename = "Model_DT_RainToday.pkl" 
with open(pkl_filename, 'rb') as file:  
    loaded_model_DT_RainToday = pickle.load(file)
    
#Load data baru yg akan diprediksi label kelasnya
df_new = pd.read_csv('new_data.csv', delimiter = ',')
#pastikan jumlah kolomnya sama dengan jumlah atribut fitur terpilih yang digunakan saat pembuatan model final
X_new = df_new[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']].values

#Lakukan prediksi (mencari nilai mpg)
Y_pred_new = loaded_model_DT_RainToday.predict(X_new)
print(Y_pred_new)
# ===================================================================================================
# Atribut kelas: 'RainTomorrow'
# Algoritma k-NN untuk k = 19

#Buat features dari kombinasi atribut terbaik yaitu  4 atribut tersignifikan
#'Rainfall', 'Sunshine', 'Humidity9am', 'Humidity3pm'
dt_weather_features = dt_weather[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']]

#Buat array Numpy untuk features
array_weather_features = np.array(dt_weather_features.values)

#Buat label kelas dari kolom 'RainTomorrow'
weather_labels = dt_weather[['RainTomorrow']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
weather_label_np = np.array(weather_labels.values) # numpy array 

#Ubah matriks 1 kolom ke 1 baris (supaya dapat menjadi parameter le.fit_transform(.))
label_np= weather_label_np.ravel()

#creating labelEncoder
le = preprocessing.LabelEncoder()

#Ubah label string ke numerik
weather_labels_en = le.fit_transform(label_np)
print(weather_labels_en)

# Import train_test_split function
from sklearn.model_selection import train_test_split

X = array_weather_features
Y = weather_labels_en

# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(array_weather_features, weather_labels_en, test_size=0.3) 

#Klasifikasi dataset wine.cvs dgn algoritma k-NN
from sklearn.neighbors import KNeighborsClassifier

#Buat model dgn jumlah neighbor = k = 19
kNN_model_weather = KNeighborsClassifier(n_neighbors=19)

#Train the model using the training sets
kNN_model_weather.fit(X_train, Y_train)

#Predict the response for test dataset
Y_pred = kNN_model_weather.predict(X_test)

weather_classes = weather_labels.RainTomorrow.unique()
print(weather_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = 'RainTomorrow'))

#====== SIMPAN MODEL ======

import pickle

#Simpan model dgn nama: Model_kNN_RainTomorrow.pkl
pkl_filename = "Model_kNN_RainTomorrow.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(kNN_model_weather, file)
    
#============== pemanfaatan model ================
    
# Cara:
#Load/baca model yg sudah disimpan
import pickle 
pkl_filename = "Model_kNN_RainTomorrow.pkl"  
with open(pkl_filename, 'rb') as file:  
    loaded_model_kNN_RainToday = pickle.load(file)


#Load data baru yg akan diprediksi label kelasnya
df_new = pd.read_csv('new_data.csv', delimiter = ',')
#pastikan jumlah kolomnya sama dengan jumlah atribut fitur pada model
X_new = df_new[['Rainfall', 'Sunshine', 'Humidity9am','Humidity3pm']].values

#Lakukan prediksi (mencari nilai mpg)
Y_pred_new = loaded_model_kNN_RainTomorrow.predict(X_new)
print(Y_pred_new)