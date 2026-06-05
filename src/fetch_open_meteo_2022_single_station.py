from pathlib import Path

import pandas as pd
import requests

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Temporary location near the Dutch coast.
# Later, this will be replaced with your selected KNMI stations.
params = {
    "latitude": 52.96,
    "longitude": 4.75,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "hourly": "wind_speed_10m,wind_direction_10m",
    "wind_speed_unit": "ms",

    # UTC is useful because your energy dataset also uses UTC timestamps.
    "timezone": "UTC",
}

response = requests.get(API_URL, params=params, timeout=60)
response.raise_for_status()

data = response.json()

if "hourly" not in data:
    raise ValueError("The API response does not contain hourly weather data.")

weather_df = pd.DataFrame(data["hourly"])

print("API connection successful.")
print(f"Number of hourly rows: {len(weather_df)}")
print(weather_df.head())
print(weather_df.tail())

output_folder = Path("data/raw")
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "open_meteo_forecast_2022_single_station.csv"
weather_df.to_csv(output_path, index=False)

print(f"\nFull-year forecast file saved to: {output_path}")