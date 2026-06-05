from pathlib import Path
import time

import pandas as pd
import requests

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

STATIONS = [
    {
        "station_id": 225,
        "station_name": "IJmuiden",
        "latitude": 52.4622,
        "longitude": 4.5550,
    },
    {
        "station_id": 242,
        "station_name": "Vlieland Vliehors",
        "latitude": 53.2400,
        "longitude": 4.9208,
    },
    {
        "station_id": 269,
        "station_name": "Lelystad Airport",
        "latitude": 52.4483,
        "longitude": 5.5081,
    },
    {
        "station_id": 310,
        "station_name": "Vlissingen",
        "latitude": 51.4414,
        "longitude": 3.5958,
    },
    {
        "station_id": 330,
        "station_name": "Hoek van Holland",
        "latitude": 51.9911,
        "longitude": 4.1217,
    },
]

OUTPUT_PATH = Path("data/raw/open_meteo_forecast_2022_all_stations.csv")


def fetch_station_data(station: dict) -> pd.DataFrame:
    """
    Retrieve archived hourly wind forecasts for one station location.
    """
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            f"No hourly data returned for station {station['station_id']}."
        )

    station_df = pd.DataFrame(data["hourly"])

    station_df["station_id"] = station["station_id"]
    station_df["station_name"] = station["station_name"]
    station_df["latitude"] = station["latitude"]
    station_df["longitude"] = station["longitude"]

    return station_df


def main() -> None:
    station_frames = []

    for station in STATIONS:
        print(
            f"Downloading station {station['station_id']}: "
            f"{station['station_name']}..."
        )

        station_df = fetch_station_data(station)

        print(f"Received {len(station_df)} hourly rows.")

        station_frames.append(station_df)

        # Small pause between requests.
        time.sleep(0.5)

    combined_df = pd.concat(station_frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDownload completed.")
    print(f"Total hourly rows: {len(combined_df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nRows per station:")
    print(combined_df.groupby(["station_id", "station_name"]).size())


if __name__ == "__main__":
    main()