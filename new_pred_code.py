# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 00:13:38 2023

@author: priya
"""
#importing the necessary libraries and dependencies
import pandas as pd
import numpy as np
import seaborn as sns;
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#%%

import datetime as dt
import calendar
from datetime import datetime
from keras_self_attention import SeqSelfAttention
from keras.layers import Dropout
from keras.layers import InputLayer


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return dt.date(year, month, day)

def extract_data(dataset, pollutants):
    # sel_columns = pollutants
    df = pd.read_csv(dataset)
    df_no_na = df.dropna(axis=0)
    cities = list(df_no_na['City'].drop_duplicates())

    filtered_data_tr = pd.DataFrame()
    filtered_data_test = pd.DataFrame()

    for city in cities:
        # Select the city
        tmp = df_no_na[df_no_na['City'] == city]

        # Pick start and end date
        start_date = list(tmp['Date'])[0]
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_tr = add_months(start_date, 3)
        end_date_test = add_months(start_date, 4)
        end_date_test = end_date_test.strftime("%Y-%m-%d")
        end_date_tr = end_date_tr.strftime("%Y-%m-%d")
        start_date = start_date.strftime("%Y-%m-%d")

        # Pick the rows that are in the selected period
        sel_dates_tr = tmp[tmp['Date'] <= end_date_tr]
        sel_dates_test = tmp[tmp['Date'] > end_date_tr]
        sel_dates_test = sel_dates_test[sel_dates_test['Date'] <= end_date_test]
        

        # Concat the dataframe of each city
        filtered_data_tr = pd.concat([filtered_data_tr, sel_dates_tr])
        filtered_data_test = pd.concat([filtered_data_test, sel_dates_test])

    pollutants_data_tr = pd.DataFrame(filtered_data_tr, columns=pollutants+['City', 'Date'])
    pollutants_labels_tr = pd.DataFrame(filtered_data_tr, columns=['City', 'Date'])
    pollutants_aqi=pd.DataFrame(filtered_data_tr, columns=['AQI','City', 'Date'])
    pollutants_data_test = pd.DataFrame(filtered_data_test, columns=pollutants+['City', 'Date'])
    pollutants_labels_test = pd.DataFrame(filtered_data_test, columns=['City', 'Date'])
    pollutants_aqi_test=pd.DataFrame(filtered_data_test, columns=['AQI','City', 'Date'])


    df_final_tr = pollutants_data_tr.merge(pollutants_labels_tr).set_index(['City', 'Date']).astype('float32')
    pollutants_aqi_final = pollutants_aqi.merge(pollutants_labels_tr).set_index(['City', 'Date']).astype('float32')
    df_final_test = pollutants_data_test.merge(pollutants_labels_test).set_index(['City', 'Date']).astype('float32')
    pollutants_aqi_test_final = pollutants_aqi_test.merge(pollutants_labels_test).set_index(['City', 'Date']).astype('float32')

    return df_final_tr, pollutants_aqi_final, df_final_test, pollutants_aqi_test_final


file=r'C:\Users\priya\Downloads\Urban-Computing-Project-main\Urban-Computing-Project-main\city_day.csv'

df_final_tr, pollutants_aqi_final, df_final_test, pollutants_aqi_test_final = extract_data(file, ['CO', 'PM2.5', 'O3', 'NH3', 'SO2', 'PM10', 'NOx'])

pollutants_data_tr=np.asarray(df_final_tr)
pollutants_aqi=np.asarray(pollutants_aqi_final)
pollutants_data_test=np.asarray(df_final_test)

#%%


#%%

#Normalizing training data
train_norm = pollutants_data_tr[:,0]

#converted into array as all the methods available are for arrays and not lists
train_norm_arr = np.asarray(train_norm)
train_norm = np.reshape(train_norm_arr, (-1, 1))

#Scaling all values between 0 and 1 so that large values don't just dominate
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(train_norm)
for i in range(5):
    print(train_norm[i])
    
count = 0
for i in range(len(train_norm)):
    if train_norm[i] == 0:
        count = count +1
print('Number of null values in train_norm = ', count)

test_norm = pollutants_data_test[:,0]
test_norm_arr = np.asarray(test_norm)
test_norm = np.reshape(test_norm_arr, (-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
test_norm = scaler.fit_transform(test_norm)
for i in range(5):
    print(test_norm[i])
    
count = 0
for i in range(len(test_norm)):
    if test_norm[i] == 0:
        count = count + 1 
print('Number of null values in test_norm = ', count)

test_norm = test_norm[test_norm != 0]

#%%

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X),array(y)

n_steps = 3
X_split_train, y_split_train = split_sequence(train_norm, n_steps)
#for i in range(len(X_split_train)):
    #print(X_split_train[i], y_split_train[i])
n_features = 1
X_split_train = X_split_train.reshape((X_split_train.shape[0], X_split_train.shape[1], n_features))
for i in range(5):
    print(X_split_train)
    
X_split_test, y_split_test = split_sequence(test_norm, n_steps)
for i in range(5):
    print(X_split_test[i], y_split_test[i])
n_features = 1
X_split_test = X_split_test.reshape((X_split_test.shape[0], X_split_test.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))

#sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=1.0, nesterov=False)
sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True) #good

#keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

hist = model.fit(X_split_train, y_split_train, validation_data=(X_split_test, y_split_test), epochs=10, verbose = 1)

yhat = model.predict(X_split_test)
for i in range(5):
    print(yhat[i])
    
mse = mean_squared_error(y_split_test, yhat)
print('MSE: %.5f' % mse)