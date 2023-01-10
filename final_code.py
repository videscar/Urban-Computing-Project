#importing the necessary libraries and dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
import calendar

from sklearn.preprocessing import MinMaxScaler
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from datetime import datetime
from cal_aqi import *
from keras.layers import Input
from keras.models import Model

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return dt.date(year, month, day)

def extract_data(dataset, pollutants):
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
        end_date_tr = add_months(start_date, 14)
        end_date_test = add_months(start_date, 15)
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


file='city_day.csv'

df_final_tr, pollutants_aqi_final, df_final_test, pollutants_aqi_test_final = extract_data(file, ['CO', 'PM2.5', 'O3', 'NH3', 'SO2', 'PM10', 'NOx'])
pollutants_data_tr=np.asarray(df_final_tr)
pollutants_aqi=np.asarray(pollutants_aqi_final)
pollutants_data_test=np.asarray(df_final_test)

#Normalizing training data
train_norm = pollutants_data_tr

#converted into array as all the methods available are for arrays and not lists
train_norm_arr = np.asarray(train_norm)

#Scaling all values between 0 and 1 so that large values don't just dominate
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(train_norm)

test_norm = pollutants_data_test
test_norm_arr = np.asarray(test_norm)
scaler = MinMaxScaler(feature_range=(0, 1))
test_norm = scaler.fit_transform(test_norm)

win_length=7
batch_size=8
X_train=train_norm_arr
y_train=train_norm_arr
X_test=test_norm_arr
y_test=test_norm_arr

train_generator = TimeseriesGenerator(X_train, y_train, length=win_length, batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test, y_test, length=win_length, batch_size=batch_size)

input = Input(shape=(win_length, 7))
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(input)
x = tf.keras.layers.Dropout(0.4)(x)
x = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, attention_activation='softmax', name='Attention')(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True))(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Conv1D(64, kernel_size=3)(x)
x = tf.keras.layers.MaxPooling1D(pool_size=1)(x)
x = tf.keras.layers.Flatten()(x)
output = tf.keras.layers.Dense(7)(x)

model_LSTM = Model(inputs=input, outputs=output)
lr_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=3, factor=0.5, cooldown=1)
model_LSTM.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), loss=tf.losses.MeanSquaredLogarithmicError(), metrics=['mse', 'mae'])

with tf.device('/GPU:0'):
    prepared_model = model_LSTM.fit_generator(train_generator, 
                                    epochs=100, 
                                    shuffle=False,
                                    callbacks=[lr_monitor])

plt.plot(prepared_model.history["loss"],label="msle")
plt.legend(loc="best")
plt.xlabel("No. Of Epochs")
plt.ylabel("metric values")
plt.show()

predictions=model_LSTM.predict(test_generator)

test_values = model_LSTM.evaluate(test_generator)
print(predictions)
pred_final=[]
for i in range(len(predictions)):
    temp=get_AQI(predictions[i])
    pred_final.append(temp[0])

test=pollutants_aqi_test_final.to_numpy()
pred_final=np.asarray(pred_final,dtype=np.float32)

plt.plot(pred_final,label = 'Predict')
plt.plot(test[win_length:],label = 'GT label')
plt.legend()
plt.show()

pollutants_aqi_test_final['AQI-PRED'] = None
pollutants_aqi_test_final['AQI-PRED'][win_length:] = pred_final

indexes = pollutants_aqi_test_final.index
cities = []
for c, _ in indexes:
    cities.append(c)
cities = [*set(cities)]
print(cities)
for city in cities:
    data = pollutants_aqi_test_final.loc[[city]]
    idx = data.index
    dates = [d for _, d in idx]
    test = data.to_numpy()
    fig = plt.figure()
    fig.suptitle(f"AQI predictions - {city}")
    plt.plot(np.asarray(data['AQI']),label = "GT label")
    plt.plot(np.asarray(data['AQI-PRED']),label = 'Predicted label')
    plt.xlabel("Nº day predicted")
    plt.ylabel("AQI values")
    plt.xticks(np.arange(len(dates)), dates, rotation=60)
    plt.legend()
    plt.show()
    # plt.savefig(f"AQI predictions - {city}.pdf")