import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import calendar
from datetime import datetime
import keras
from keras.models import Model
from keras.layers import Dense, Conv1D
from keras_self_attention import SeqSelfAttention
from keras.layers import Input

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

inputs = Input(shape=(7,1))

x = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL, attention_activation='softmax', name='Attention')(inputs)

x = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True))(x)

x =  Conv1D(128, kernel_size=3, activation='relu')(x)

x =  Conv1D(64, kernel_size=3, activation='relu')(x)

x =  Conv1D(32, kernel_size=3, activation='relu')(x)

x =  Conv1D(16, kernel_size=1, activation='relu')(x)

x = Dense(34 ,'relu')(x)

x = Dense(34 ,'relu')(x)

x = Dense(15 ,'relu')(x)

x = Dense(15 ,'relu')(x)

output = Dense(1)(x)

model_lstm = Model(inputs=inputs, outputs=output)

model_lstm.compile(optimizer = 'adam', loss = 'mae', metrics = ['mape'])

lstm_history = model_lstm.fit(pollutants_data_tr, pollutants_aqi, validation_split=0.2 , epochs = 250, shuffle=False)
lstm_y_pred = model_lstm.predict(pollutants_data_test)

pred_final=[]
for i in range(len(lstm_y_pred)):
    temp=np.mean(lstm_y_pred[i])
    pred_final.append(temp)

pollutants_aqi_test_final["AQI-PRED"] = pred_final

indexes = pollutants_aqi_test_final.index
cities = []
for c, _ in indexes:
    cities.append(c)
cities = [*set(cities)]

for city in cities:
    data = pollutants_aqi_test_final.loc[[city]]
    idx = data.index
    dates = [d for _, d in idx]
    fig = plt.figure()
    fig.suptitle(f"AQI predictions - {city}")
    plt.plot(np.asarray(data['AQI']),label = "Test")
    plt.plot(np.asarray(data['AQI-PRED']),label = 'Predict')
    plt.xticks(np.arange(len(dates)), dates, rotation=60)
    plt.legend()
    # plt.show()
    plt.savefig(f"AQI predictions - {city}.pdf")
