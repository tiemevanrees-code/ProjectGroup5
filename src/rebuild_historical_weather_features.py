from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

KNMI_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "wind_data"
    / "KNMI_cleaned_mvp.csv"
)

CURRENT_MERGED_PATH = (
    PROJECT_ROOT
    / "MVP"
    / "merged_weather_energy_daily_mvp.csv"
)

OUTPUT_PATH = CURRENT_MERGED_PATH

HISTORICAL_START_DATE = pd.Timestamp("2017-01-01")
HISTORICAL_END_DATE = pd.Timestamp("2021-12-31")


def load_historical_energy_target() -> pd.DataFrame:
    """
    Keep the existing historical wind-energy target unchanged.

    This script only rebuilds the weather features so they are calculated
    consistently for future model comparisons.
    """
    if not CURRENT_MERGED_PATH.exists():
        raise FileNotFoundError(
            f"Existing merged dataset not found: {CURRENT_MERGED_PATH}"
        )

    merged_df = pd.read_csv(CURRENT_MERGED_PATH)

    required_columns = {
        "date",
        "daily_wind_generation_kwh",
    }

    missing_columns = required_columns - set(merged_df.columns)

    if missing_columns:
        raise ValueError(
            "Existing merged dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    energy_df = merged_df[
        [
            "date",
            "daily_wind_generation_kwh",
        ]
    ].copy()

    energy_df["date"] = pd.to_datetime(
        energy_df["date"],
        errors="raise",
    )

    energy_df = energy_df[
        energy_df["date"].between(
            HISTORICAL_START_DATE,
            HISTORICAL_END_DATE,
        )
    ].copy()

    if energy_df["date"].duplicated().any():
        duplicate_dates = energy_df[
            energy_df["date"].duplicated(keep=False)
        ]

        raise ValueError(
            "Existing energy target contains duplicate dates:\n"
            f"{duplicate_dates}"
        )

    if energy_df["daily_wind_generation_kwh"].isna().any():
        raise ValueError(
            "Existing energy target contains missing values."
        )

    return energy_df


def build_historical_weather_features() -> pd.DataFrame:
    """
    Recalculate daily historical weather features consistently.

    Wind direction is circular data. Therefore, direction cannot be
    averaged using a normal arithmetic mean. Instead, station-level wind
    vectors are converted into horizontal components first. The daily
    vector-average wind speed and circular average direction are then
    derived from the averaged components.
    """
    if not KNMI_PATH.exists():
        raise FileNotFoundError(
            f"KNMI dataset not found: {KNMI_PATH}"
        )

    df = pd.read_csv(KNMI_PATH)

    required_columns = {
        "station_id",
        "date",
        "wind_direction_deg",
        "vector_avg_wind_speed_ms",
        "avg_wind_speed_ms",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "KNMI dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    # Keep only the intended historical model period.
    # The source file contains partial boundary dates outside this range.
    df = df[
        df["date"].between(
            HISTORICAL_START_DATE,
            HISTORICAL_END_DATE,
        )
    ].copy()

    numeric_columns = [
        "station_id",
        "wind_direction_deg",
        "vector_avg_wind_speed_ms",
        "avg_wind_speed_ms",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # These values are required for vector calculations.
    incomplete_rows = df[
        df[
            [
                "wind_direction_deg",
                "vector_avg_wind_speed_ms",
                "avg_wind_speed_ms",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    if not incomplete_rows.empty:
        incomplete_dates = (
            incomplete_rows["date"]
            .drop_duplicates()
            .sort_values()
        )

        print("Removing incomplete KNMI dates:")

        for date in incomplete_dates:
            print(f"- {date.date()}")

        # Remove the complete day so every retained day uses the same
        # station coverage.
        df = df[
            ~df["date"].isin(incomplete_dates)
        ].copy()

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "KNMI dataset still contains missing numeric values."
        )

    direction_radians = np.radians(
        df["wind_direction_deg"]
    )

    # Convert station-level vector wind measurements into components.
    # This follows the meteorological convention for wind direction.
    df["wind_u_ms"] = (
        -df["vector_avg_wind_speed_ms"]
        * np.sin(direction_radians)
    )

    df["wind_v_ms"] = (
        -df["vector_avg_wind_speed_ms"]
        * np.cos(direction_radians)
    )

    daily_df = (
        df.groupby("date")
        .agg(
            avg_wind_speed_all_stations=(
                "avg_wind_speed_ms",
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
        )
        .reset_index()
    )

    invalid_station_days = daily_df[
        daily_df["station_count"] != 5
    ]

    if not invalid_station_days.empty:
        raise ValueError(
            "Some dates do not contain all five stations:\n"
            f"{invalid_station_days}"
        )

    # Length of the daily averaged wind vector.
    daily_df["vector_avg_wind_speed_all_stations"] = np.sqrt(
        daily_df["wind_u_mean"] ** 2
        + daily_df["wind_v_mean"] ** 2
    )

    # Circular average wind direction in degrees.
    daily_df["avg_wind_direction"] = (
        np.degrees(
            np.arctan2(
                -daily_df["wind_u_mean"],
                -daily_df["wind_v_mean"],
            )
        )
        + 360
    ) % 360

    # Existing physically motivated non-linear feature.
    daily_df["avg_wind_speed_cubed"] = (
        daily_df["avg_wind_speed_all_stations"] ** 3
    )

    # Model-ready circular direction features.
    daily_direction_radians = np.radians(
        daily_df["avg_wind_direction"]
    )

    daily_df["wind_direction_sin"] = np.sin(
        daily_direction_radians
    )

    daily_df["wind_direction_cos"] = np.cos(
        daily_direction_radians
    )

    daily_df = daily_df.drop(
        columns=[
            "wind_u_mean",
            "wind_v_mean",
        ]
    )

    return daily_df


def merge_and_validate(
    weather_df: pd.DataFrame,
    energy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge rebuilt historical weather features with the existing target.
    """
    merged_df = pd.merge(
        weather_df,
        energy_df,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    merged_df = (
        merged_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    if merged_df.isna().any().any():
        raise ValueError(
            "Merged historical dataset contains missing values."
        )

    if merged_df["date"].duplicated().any():
        duplicate_dates = merged_df[
            merged_df["date"].duplicated(keep=False)
        ]

        raise ValueError(
            "Merged historical dataset contains duplicate dates:\n"
            f"{duplicate_dates}"
        )

    return merged_df


def main() -> None:
    energy_df = load_historical_energy_target()
    weather_df = build_historical_weather_features()

    merged_df = merge_and_validate(
        weather_df,
        energy_df,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nHistorical weather features rebuilt successfully.")
    print(f"Rows: {len(merged_df)}")
    print(
        "Date range:",
        merged_df["date"].min().date(),
        "to",
        merged_df["date"].max().date(),
    )

    print("\nColumns:")
    for column in merged_df.columns:
        print(f"- {column}")

    print("\nMissing values:")
    print(merged_df.isna().sum())

    print("\nFirst five rows:")
    print(merged_df.head())


if __name__ == "__main__":
    main()