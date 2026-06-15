from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "processed"
    / "knmi_actual_2022_daily.csv"
)

STATION_OUTPUT_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "processed"
    / "knmi_actual_2022_station_daily.csv"
)

KNMI_URL = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

# Same stations as your 2017-2021 KNMI dataset
STATIONS = [225, 242, 269, 310, 330]

WIND_VARIABLES = [
    "DDVEC",  # vector-average wind direction in degrees
    "FHVEC",  # vector-average wind speed in 0.1 m/s
    "FG",     # daily average wind speed in 0.1 m/s
    "FHX",    # highest hourly average wind speed in 0.1 m/s
    "FHXH",   # hour of highest hourly average wind speed
    "FHN",    # lowest hourly average wind speed in 0.1 m/s
    "FHNH",   # hour of lowest hourly average wind speed
]


def download_knmi_daily_data() -> str:
    response = requests.post(
        KNMI_URL,
        data={
            "stns": ":".join(str(station) for station in STATIONS),
            "vars": ":".join(WIND_VARIABLES),
            "start": "20220101",
            "end": "20221231",
        },
        timeout=60,
    )

    response.raise_for_status()
    return response.text


def parse_knmi_response(raw_text: str) -> pd.DataFrame:
    data_lines = []

    for line in raw_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue

        data_lines.append(line)

    csv_text = "\n".join(data_lines)

    df = pd.read_csv(
        StringIO(csv_text),
        header=None,
        names=[
            "station_id",
            "date",
            "wind_direction_deg",
            "vector_avg_wind_speed_ms",
            "avg_wind_speed_ms",
            "max_wind_speed_ms",
            "max_wind_hour",
            "min_wind_speed_ms",
            "min_wind_hour",
        ],
        skipinitialspace=True,
    )

    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    wind_speed_columns = [
        "vector_avg_wind_speed_ms",
        "avg_wind_speed_ms",
        "max_wind_speed_ms",
        "min_wind_speed_ms",
    ]

    for column in wind_speed_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce") / 10

    df["wind_direction_deg"] = pd.to_numeric(
        df["wind_direction_deg"],
        errors="coerce",
    )

    df["max_wind_hour"] = pd.to_numeric(df["max_wind_hour"], errors="coerce")
    df["min_wind_hour"] = pd.to_numeric(df["min_wind_hour"], errors="coerce")

    df = df.dropna(
        subset=[
            "date",
            "wind_direction_deg",
            "vector_avg_wind_speed_ms",
            "avg_wind_speed_ms",
        ]
    )

    return df


def create_daily_features(station_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = (
        station_df
        .groupby("date")
        .agg(
            avg_wind_speed_all_stations=("avg_wind_speed_ms", "mean"),
            vector_avg_wind_speed_all_stations=("vector_avg_wind_speed_ms", "mean"),
            avg_wind_direction=("wind_direction_deg", "mean"),
            station_count=("station_id", "nunique"),
        )
        .reset_index()
    )

    daily_df["avg_wind_speed_cubed"] = (
        daily_df["avg_wind_speed_all_stations"] ** 3
    )

    direction_radians = np.deg2rad(daily_df["avg_wind_direction"])

    daily_df["wind_direction_sin"] = np.sin(direction_radians)
    daily_df["wind_direction_cos"] = np.cos(direction_radians)

    daily_df["date"] = daily_df["date"].dt.strftime("%Y-%m-%d")

    return daily_df


def main() -> None:
    raw_text = download_knmi_daily_data()

    station_df = parse_knmi_response(raw_text)
    daily_df = create_daily_features(station_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    station_df.to_csv(STATION_OUTPUT_PATH, index=False)
    daily_df.to_csv(OUTPUT_PATH, index=False)

    print("Actual KNMI 2022 wind data created successfully.")
    print(f"Station-level file: {STATION_OUTPUT_PATH}")
    print(f"Daily feature file: {OUTPUT_PATH}")
    print(f"Rows in daily file: {len(daily_df)}")
    print()
    print(daily_df.head())


if __name__ == "__main__":
    main()