'''
NOT IMPORTANT YOU CAN IGNORE THIS FILE
i just wanted to test the open-meteo API
'''
import requests
import pandas as pd

latitude = 38.65
longitude = 23.9

# start_date = "2015-05-01"
start_date = "2025-02-28"
end_date   = "2025-02-28"

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
resp = requests.get(url)
resp.raise_for_status()

data = resp.json()
df = pd.DataFrame(data['daily'])

output_file = "greece_weather_2015_2025.csv"
df.to_csv(output_file, index=False)
print(f"✅ Saved weather data ({len(df)} days) to: {output_file}")
