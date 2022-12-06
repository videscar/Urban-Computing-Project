import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import tensorflow as tf
import calendar
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Conv1D, Input, Dropout, AvgPool1D, Reshape, Concatenate,Flatten

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


file='city_day.csv'

df_final_tr, pollutants_aqi_final, df_final_test, pollutants_aqi_test_final = extract_data(file, ['CO', 'PM2.5', 'O3', 'NH3', 'SO2', 'PM10', 'NOx'])

pollutants_data_tr=np.asarray(df_final_tr)
pollutants_aqi=np.asarray(pollutants_aqi_final)
pollutants_data_test=np.asarray(df_final_test)
# pollutants_aqi_test_final=np.asarray(pollutants_aqi_test_final)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import InputLayer
from keras import backend as K

### a combination of LSTM and CNN for boosted performance

inputs = Input(shape=(7,1))

### top pipeline

top_lstm = LSTM(512)(inputs)
top_dense = Dense(512, activation='relu')(top_lstm)
# top_dropout = Dropout(256)(top_dense)


### bottom pipeline

bottom_dense = Dense(512)(top_dense[:,:,np.newaxis])
bottom_conv1 = Conv1D(
    256, 
    kernel_size=3,
    activation='relu'
)(bottom_dense)
bottom_conv2 = Conv1D(
    512,
    kernel_size=5,
    padding='same',
    activation='relu'
)(bottom_conv1)
bottom_conv3 = Conv1D(
    256,
    kernel_size=3,
    padding='same',
    activation='relu'
)(bottom_conv2)
bottom_pooling = AvgPool1D(
    pool_size=32, 
    padding='same'
)(bottom_conv3)
bottom_reshape = Flatten()(bottom_conv3)


### concatenate output from both pipelines

# final_concat = Concatenate()([top_dropout, bottom_reshape])
first_dense = Dense(256)(bottom_reshape)
final_dense = Dense(1)(first_dense)
# compile and return

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

complex_model = Model(inputs=inputs, outputs=final_dense)

# lr_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.005, cooldown=1)
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

complex_model.compile(loss="mae", optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001))

complex_history = complex_model.fit(
    pollutants_data_tr, 
    pollutants_aqi, 
    epochs=100, 
    batch_size=128,
    validation_split=0.2,
    verbose=2,
    shuffle=False
)


# Plot the result in train & validation
# hist = pd.DataFrame(complex_history.history)
# pd.DataFrame(hist).plot(figsize=(8, 5))
# plt.grid(True)
# plt.show()

test_eval = complex_model.evaluate(pollutants_data_test, pollutants_aqi_test_final)

lstm_y_pred = complex_model.predict(pollutants_data_test)

pollutants_aqi_test_final["AQI-PRED"] = lstm_y_pred

indexes = pollutants_aqi_test_final.index
cities = []
for c, _ in indexes:
    cities.append(c)
cities = [*set(cities)]

for city in cities:
    data = pollutants_aqi_test_final.loc[[city]]
    idx = data.index
    dates = [d for _, d in idx]
    fig = plt.figure(figsize=(8,25))
    fig.suptitle(f"AQI predictions - {city}")
    plt.plot(np.asarray(data['AQI']),label = "Test")
    plt.plot(np.asarray(data['AQI-PRED']),label = 'Predict')
    plt.xticks(np.arange(len(dates)), dates, rotation=60)
    plt.legend()
    plt.show()
