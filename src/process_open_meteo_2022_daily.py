from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/raw/open_meteo_forecast_2022_all_stations.csv")
OUTPUT_PATH = Path("data/processed/open_meteo_forecast_2022_daily.csv")


def load_hourly_data() -> pd.DataFrame:
    """
    Load the hourly forecast data downloaded from Open-Meteo.
    """
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run fetch_open_meteo_2022_all_stations.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "time",
        "wind_speed_10m",
        "wind_direction_10m",
        "station_id",
        "station_name",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["datetime"].dt.date

    return df


def add_wind_vector_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wind speed and direction into vector components.

    This avoids calculating an incorrect ordinary average for wind
    direction. For example, 359 degrees and 1 degree are both close
    to north, but their ordinary numerical average would be 180 degrees.
    """
    direction_radians = np.radians(df["wind_direction_10m"])

    df["wind_u_ms"] = -df["wind_speed_10m"] * np.sin(direction_radians)
    df["wind_v_ms"] = -df["wind_speed_10m"] * np.cos(direction_radians)

    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all hourly station forecasts into one row per day.
    """
    daily_df = (
        df.groupby("date")
        .agg(
            avg_wind_speed_all_stations=(
                "wind_speed_10m",
                "mean",
            ),
            wind_u_mean=(
                "wind_u_ms",
                "mean",
            ),
            wind_v_mean=(
                "wind_v_ms",
                "mean",
            ),
            station_count=(
                "station_id",
                "nunique",
            ),
            hourly_record_count=(
                "wind_speed_10m",
                "count",
            ),
        )
        .reset_index()
    )

    # Vector-average wind speed
    daily_df["vector_avg_wind_speed_all_stations"] = np.sqrt(
        daily_df["wind_u_mean"] ** 2
        + daily_df["wind_v_mean"] ** 2
    )

    # Circular average wind direction
    daily_df["avg_wind_direction"] = (
        np.degrees(
            np.arctan2(
                -daily_df["wind_u_mean"],
                -daily_df["wind_v_mean"],
            )
        )
        + 360
    ) % 360

    # Physically motivated engineered feature
    daily_df["avg_wind_speed_cubed"] = (
        daily_df["avg_wind_speed_all_stations"] ** 3
    )

    # ML-ready circular direction features
    direction_radians = np.radians(daily_df["avg_wind_direction"])

    daily_df["wind_direction_sin"] = np.sin(direction_radians)
    daily_df["wind_direction_cos"] = np.cos(direction_radians)

    # Remove intermediate calculation columns
    daily_df = daily_df.drop(
        columns=[
            "wind_u_mean",
            "wind_v_mean",
        ]
    )

    return daily_df


def validate_daily_data(daily_df: pd.DataFrame) -> None:
    """
    Check whether the daily result is complete and suitable for modeling.
    """
    expected_days = 365
    expected_records_per_day = 5 * 24

    if len(daily_df) != expected_days:
        raise ValueError(
            f"Expected {expected_days} daily rows, "
            f"but received {len(daily_df)}."
        )

    if daily_df.isna().any().any():
        raise ValueError("The daily dataset contains missing values.")

    if not (daily_df["station_count"] == 5).all():
        raise ValueError("Not every day contains all five stations.")

    if not (
        daily_df["hourly_record_count"] == expected_records_per_day
    ).all():
        raise ValueError(
            "Not every day contains 120 hourly station records."
        )


def main() -> None:
    hourly_df = load_hourly_data()
    hourly_df = add_wind_vector_components(hourly_df)

    daily_df = aggregate_to_daily(hourly_df)
    validate_daily_data(daily_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(OUTPUT_PATH, index=False)

    print("Daily processing completed successfully.")
    print(f"Number of daily rows: {len(daily_df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst five rows:")
    print(daily_df.head())

    print("\nValidation summary:")
    print(daily_df[["station_count", "hourly_record_count"]].describe())


if __name__ == "__main__":
    main()