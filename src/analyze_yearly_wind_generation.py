from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

HISTORICAL_PATH = (
    PROJECT_ROOT
    / "MVP"
    / "merged_weather_energy_daily_mvp.csv"
)

ACTUAL_2022_PATH = (
    SRC_DIR
    / "data"
    / "True_power_generation_in_2022"
    / "wind-2022-uur-data-reformated.csv"
)

OUTPUT_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "yearly_wind_generation_summary.csv"
)

HISTORICAL_TARGET_COLUMN = (
    "daily_wind_generation_kwh"
)

ACTUAL_2022_TARGET_COLUMN = (
    "Actual Energy created (kWh)"
)


def load_historical_data() -> pd.DataFrame:
    """
    Load historical daily wind-generation values for 2017-2021.
    """
    if not HISTORICAL_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: "
            f"{HISTORICAL_PATH}"
        )

    df = pd.read_csv(HISTORICAL_PATH)

    required_columns = {
        "date",
        HISTORICAL_TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Historical dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    return (
        df[
            [
                "date",
                HISTORICAL_TARGET_COLUMN,
            ]
        ]
        .rename(
            columns={
                HISTORICAL_TARGET_COLUMN:
                    "daily_wind_generation_kwh"
            }
        )
    )


def load_actual_2022_data() -> pd.DataFrame:
    """
    Load actual daily wind-generation values for 2022.
    """
    if not ACTUAL_2022_PATH.exists():
        raise FileNotFoundError(
            f"2022 dataset not found: "
            f"{ACTUAL_2022_PATH}"
        )

    df = pd.read_csv(ACTUAL_2022_PATH)

    required_columns = {
        "date",
        ACTUAL_2022_TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "2022 dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    return (
        df[
            [
                "date",
                ACTUAL_2022_TARGET_COLUMN,
            ]
        ]
        .rename(
            columns={
                ACTUAL_2022_TARGET_COLUMN:
                    "daily_wind_generation_kwh"
            }
        )
    )


def calculate_percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    """
    Calculate percentage change from an older value
    to a newer value.
    """
    return (
        (new_value - old_value)
        / old_value
    ) * 100


def main() -> None:
    historical_df = load_historical_data()

    actual_2022_df = load_actual_2022_data()

    combined_df = pd.concat(
        [
            historical_df,
            actual_2022_df,
        ],
        ignore_index=True,
    )

    combined_df["year"] = (
        combined_df["date"].dt.year
    )

    yearly_summary = (
        combined_df
        .groupby("year")[
            "daily_wind_generation_kwh"
        ]
        .agg(
            number_of_days="count",
            average_daily_generation_kwh="mean",
            median_daily_generation_kwh="median",
            minimum_daily_generation_kwh="min",
            maximum_daily_generation_kwh="max",
        )
        .reset_index()
    )

    yearly_summary[
        "average_change_from_previous_year_percent"
    ] = (
        yearly_summary[
            "average_daily_generation_kwh"
        ]
        .pct_change()
        * 100
    )

    yearly_summary[
        "median_change_from_previous_year_percent"
    ] = (
        yearly_summary[
            "median_daily_generation_kwh"
        ]
        .pct_change()
        * 100
    )

    average_2021 = yearly_summary.loc[
        yearly_summary["year"] == 2021,
        "average_daily_generation_kwh",
    ].iloc[0]

    average_2022 = yearly_summary.loc[
        yearly_summary["year"] == 2022,
        "average_daily_generation_kwh",
    ].iloc[0]

    median_2021 = yearly_summary.loc[
        yearly_summary["year"] == 2021,
        "median_daily_generation_kwh",
    ].iloc[0]

    median_2022 = yearly_summary.loc[
        yearly_summary["year"] == 2022,
        "median_daily_generation_kwh",
    ].iloc[0]

    average_increase_2021_to_2022 = (
        calculate_percentage_change(
            average_2021,
            average_2022,
        )
    )

    median_increase_2021_to_2022 = (
        calculate_percentage_change(
            median_2021,
            median_2022,
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    yearly_summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nYearly wind-generation summary"
    )

    print("-" * 120)

    print(
        yearly_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:,.2f}"
            ),
        )
    )

    print(
        "\n2021 versus 2022 comparison"
    )

    print("-" * 72)

    print(
        "Average daily generation increase: "
        f"{average_increase_2021_to_2022:.2f}%"
    )

    print(
        "Median daily generation increase:  "
        f"{median_increase_2021_to_2022:.2f}%"
    )

    print(
        f"\nSummary saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()