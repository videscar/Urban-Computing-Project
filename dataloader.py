import pandas as pd
import matplotlib as plt
import numpy as np
import datetime as dt
import calendar
from datetime import datetime

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
        end_date_tr = add_months(start_date, 2)
        end_date_test = add_months(start_date, 3)
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

    pollutants_data_tr = pd.DataFrame(filtered_data_tr, columns=pollutants)
    pollutants_labels_tr = pd.DataFrame(filtered_data_tr, columns=['City', 'Date'])

    pollutants_data_test = pd.DataFrame(filtered_data_test, columns=pollutants)
    pollutants_labels_test = pd.DataFrame(filtered_data_test, columns=['City', 'Date'])

    print("######## TRAINING: ########")
    print(pollutants_data_tr)
    print(pollutants_labels_tr)

    print("######## TESt: ########")
    print(pollutants_data_test)
    print(pollutants_labels_test)

extract_data("city_day.csv", ['CO', 'PM2.5', 'O3', 'NO2', 'SO2', 'PM10', 'NO'])

