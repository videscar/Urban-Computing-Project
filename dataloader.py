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
    sel_columns = pollutants
    df = pd.read_csv(dataset)
    df_no_na = df.dropna(axis=0)
    cities = list(df_no_na['City'].drop_duplicates())

    filtered_data = pd.DataFrame()

    for city in cities:
        tmp = df_no_na[df_no_na['City'] == city]
        start_date = list(tmp['Date'])[0]
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = add_months(start_date, 2)
        end_date = end_date.strftime("%Y-%m-%d")
        start_date = start_date.strftime("%Y-%m-%d")
        sel_dates = tmp[tmp['Date'] <= end_date]
        filtered_data = pd.concat([filtered_data, sel_dates])

    pollutants_data = pd.DataFrame(filtered_data, columns=pollutants)
    pollutants_labels = pd.DataFrame(filtered_data, columns=['City', 'Date'])

    print(pollutants_data)
    print(pollutants_labels)



extract_data("city_day.csv", ['CO', 'PM2.5', 'O3', 'NO2', 'SO2', 'PM10', 'NO'])

