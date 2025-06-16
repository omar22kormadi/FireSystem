import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from meteostat import Point
from meteostat import Daily
import pandas as pd
from datetime import datetime
from requests.exceptions import ReadTimeout, ConnectionError

import os
def load_data(file_path):
    return pd.read_csv(file_path , sep = ";" )

def preprocess_data(df):    
    # remove feature satelite
    df.drop(columns=['satellite','version','instrument'], inplace=True)
    
    return df
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test 

def getWeather(dff):
    for long, lat, date in zip(dff['longitude'], dff['latitude'], dff['acq_date']):
        print(f"Longitude: {long}, Latitude: {lat}, Date: {date}")

        latitude = lat
        longitude = long
        start_date = date
        end_date = date

        daily_vars = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "relative_humidity_2m_max"
        ]

        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}&longitude={longitude}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&daily={','.join(daily_vars)}"
            "&timezone=Europe%2FAthens"
        )

        print("🔗 Fetching data from:", url)
        tries = 3
        for attempt in range(tries):
            try:
                resp = requests.get(url, timeout=15)  # 15 seconds timeout
                resp.raise_for_status()
                data = resp.json()
                df = pd.DataFrame(data['daily'])
                df['latitude'] = latitude
                df['longitude'] = longitude
                output_file = "greece_weather_2019_2025.csv"
                df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
                print(f"✅ Saved weather data ({len(df)} days) to: {output_file}")
                break  # Success, exit retry loop
            except (ReadTimeout, ConnectionError) as e:
                print(f"⚠️ Timeout or connection error: {e}. Retrying ({attempt+1}/{tries})...")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                print(f"❌ Failed to fetch data: {e}")
                break
        time.sleep(1)  # polite delay between requests
        
def weather(dff):
    # features : time,tavg,tmin,tmax,prcp,snow,wdir,wspd,wpgt,pres,tsun,latitude,longitude

    for long, lat, date in zip(dff['longitude'], dff['latitude'], dff['acq_date']):
        year, month, day = date.split('-')
        location = Point(lat, long)
        start = datetime(int(year), int(month), int(day))
        end = datetime(int(year), int(month), int(day))

        try:
            data = Daily(location, start, end)
            data = data.fetch()

            if not data.empty:
                data.reset_index(inplace=True)
                data['latitude'] = lat
                data['longitude'] = long
                data.to_csv("evia_meteostat_weather.csv", index=False, mode='a',
                            header=not os.path.exists("evia_meteostat_weather.csv"))
                print(f"✅ Saved weather for {date} at ({lat}, {long})")
            else:
                print(f"⚠️ No data found for {date} at ({lat}, {long})")
                data = pd.DataFrame(columns=['time', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun', 'latitude', 'longitude'] , index= [0])
                data['time'] = date
                data['latitude'] = lat
                data['longitude'] = long
               
                    
                                   
                
                data.to_csv("evia_meteostat_weather.csv", index=False, mode='a',header=not os.path.exists("evia_meteostat_weather.csv"))

        except Exception as e:
            print(f"❌ Error fetching weather for {date}: {e}")
# Example usage:    

def clean_data(file_path):
    df = load_data(file_path)
    print(df.head())
    columns_to_keep = [ 'latitude', 'longitude', 'brightness','scan','track', 'acq_date','confidence', 'bright_t31','frp','daynight','type','tavg', 'tmin', 'tmax', 'prcp', 'wspd']
    df = df[columns_to_keep]
    print(df.head())
    df.fillna({
        # 'tavg': 0,
        # 'tmin': 0,
        # 'tmax': 0,
        'prcp': 0,
        # 'wdir': 0,
        # 'wspd': 0,
        # 'pres': 0
    }, inplace=True)
    df = df.dropna()
    print("-------------------")
    #sort by acq_date
    df['acq_date'] = pd.to_datetime(df['acq_date'], format='%d/%m/%Y')

    
    # df.sort_values(by='acq_date', inplace=True)
    print(df.head())
    df.to_csv("cleaned.csv", mode='a',header=not os.path.exists("cleaned.csv"), index=False)
    return df

# df_fire = load_data('fire_data.csv')
# df_fire_prep = preprocess_data(df_fire)
# print(df_fire_prep.head())
# getWeather(dff)
# weather(df_fire_prep)
final = clean_data('fire.csv')



