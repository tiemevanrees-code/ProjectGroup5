from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

SRC_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "baseline_linear_regression_predictions_2022.csv"
)

OUTPUT_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "baseline_linear_regression_predictions_2022.png"
)

PREDICTION_COLUMN = "predicted_wind_generation_kwh"


def millions_formatter(value: float, position: int) -> str:
    """
    Format the y-axis values as millions of kWh.
    """
    return f"{value / 1_000_000:.0f}M"


def load_predictions() -> pd.DataFrame:
    """
    Load and validate the baseline prediction results.
    """
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}\n"
            "Run train_baseline_linear_regression.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "date",
        "avg_wind_speed_all_stations",
        PREDICTION_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df.isna().any().any():
        raise ValueError("The predictions dataset contains missing values.")

    if df["date"].duplicated().any():
        raise ValueError("The predictions dataset contains duplicate dates.")

    return df.sort_values("date").reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    """
    Print useful checks for the generated predictions.
    """
    zero_predictions = (df[PREDICTION_COLUMN] == 0).sum()

    print("Baseline prediction summary")
    print("-" * 60)
    print(f"Number of daily predictions: {len(df)}")
    print(
        "Minimum prediction:        "
        f"{df[PREDICTION_COLUMN].min():,.2f} kWh"
    )
    print(
        "Maximum prediction:        "
        f"{df[PREDICTION_COLUMN].max():,.2f} kWh"
    )
    print(
        "Average prediction:        "
        f"{df[PREDICTION_COLUMN].mean():,.2f} kWh"
    )
    print(f"Predictions clipped to zero: {zero_predictions}")


def create_graph(df: pd.DataFrame) -> None:
    """
    Create a daily prediction graph with a rolling-average trend line.
    """
    df["rolling_average_30_days"] = (
        df[PREDICTION_COLUMN]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df[PREDICTION_COLUMN],
        label="Daily predicted wind energy",
        alpha=0.55,
    )

    plt.plot(
        df["date"],
        df["rolling_average_30_days"],
        label="30-day rolling average",
        linewidth=2.5,
    )

    plt.title(
        "Baseline Prediction of Daily Wind-Energy Generation in 2022"
    )

    plt.xlabel("Date")
    plt.ylabel("Predicted wind-energy generation (kWh)")

    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(millions_formatter)
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"\nGraph saved to: {OUTPUT_PATH}")


def main() -> None:
    predictions_df = load_predictions()

    print_summary(predictions_df)
    create_graph(predictions_df)


if __name__ == "__main__":
    main()