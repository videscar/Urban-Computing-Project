import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import calendar
from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Conv1D, Input, Dropout, AvgPool1D, Reshape, Concatenate,Flatten
import keras
from keras_self_attention import SeqSelfAttention
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import InputLayer


#%%
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

model_lstm = Sequential()

model_lstm.add(InputLayer((7,1)))



model_lstm.add(SeqSelfAttention(
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation='softmax',
        name='Attention'))


# model_lstm.add(Dense(34 ,'relu'))
# # model_lstm.add(Dropout(0.25))


# # model_lstm.add(LSTM(50))
# model_lstm.add(Dense(34 ,'relu'))
model_lstm.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                        return_sequences=True)))
model_lstm.add(Dense(34 ,'relu'))
# model_lstm.add(Dropout(0.25))


# model_lstm.add(LSTM(50))
model_lstm.add(Dense(34 ,'relu'))

model_lstm.add(Dense(15 ,'relu'))

model_lstm.add(Dense(1 ,'relu' ))


model_lstm.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mape']
)


lstm_history = model_lstm.fit(pollutants_data_tr,pollutants_aqi,validation_split=0.33 , epochs = 70)
lstm_y_pred = model_lstm.predict(pollutants_data_test)
pred_final=[]
for i in range(len(lstm_y_pred)):
    temp=np.mean(lstm_y_pred[i])
    pred_final.append(temp)
    

# plt.figure(figsize = (20,10))
# plt.plot(pollutants_data_tr[-100:,:,0,0],pollutants_aqi[-100:],label = "Train")
# plt.plot(pollutants_data_test[:100],pollutants_aqi_test[:100],label = "Test")
# plt.plot(pollutants_data_test[:100],lstm_y_pred[:100],label = 'Predict')
# plt.legend()
# plt.show()

plt.plot(pollutants_aqi[-100:],label = "Train")
plt.plot(np.asarray(pollutants_aqi_test_final[:100]),label = "Test")
plt.plot(np.asarray(lstm_y_pred[:100,4,:]),label = 'Predict')
plt.plot(np.asarray(pred_final[:100]),label = 'Predict_final')
plt.legend()
plt.show()


