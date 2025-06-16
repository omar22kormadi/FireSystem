'''
NOT IMPORTANT YOU CAN IGNORE THIS FILE
i just wanted to test the meteostat library
'''

import pandas as pd
from datetime import datetime
from meteostat import Point
from meteostat import Daily
import os

lat = 33.7817164
long = 11.0451637
location = Point(lat, long)
start = datetime(2025, 6, 12)
end = datetime(2025, 6, 18)

try:
    data = Daily(location, start, end)
    data = data.fetch()

    if not data.empty:
       print(f"✅ Successfully fetched weather data for {start.date()} to {end.date()} at ({lat}, {long})")
    else:
         print(f"⚠️ No data found for {start.date()} to {end.date()} at ({lat}, {long})")
        
            
                            
        
    data.to_csv("test.csv", index=True, mode='a',header=not os.path.exists("test.csv"))

except Exception as e:
    print(f"❌ Error fetching weather: {e}")