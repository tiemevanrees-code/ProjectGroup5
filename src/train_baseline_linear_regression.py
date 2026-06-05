from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

HISTORICAL_PATH = (
    PROJECT_ROOT / "MVP" / "merged_weather_energy_daily_mvp.csv"
)

FORECAST_2022_PATH = (
    SRC_DIR / "data" / "processed" / "open_meteo_forecast_2022_daily.csv"
)

OUTPUT_FOLDER = SRC_DIR / "data" / "model_output"

PREDICTIONS_OUTPUT_PATH = (
    OUTPUT_FOLDER / "baseline_linear_regression_predictions_2022.csv"
)

MODEL_OUTPUT_PATH = (
    OUTPUT_FOLDER / "baseline_linear_regression.joblib"
)

FEATURE_COLUMN = "avg_wind_speed_all_stations"
TARGET_COLUMN = "daily_wind_generation_kwh"


def load_historical_data() -> pd.DataFrame:
    """
    Load and validate the historical 2017-2021 weather-energy dataset.
    """
    if not HISTORICAL_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {HISTORICAL_PATH}"
        )

    df = pd.read_csv(HISTORICAL_PATH)

    required_columns = {
        "date",
        FEATURE_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Historical dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df[[FEATURE_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError("Historical dataset contains missing values.")

    if df["date"].duplicated().any():
        raise ValueError("Historical dataset contains duplicate dates.")

    return df.sort_values("date").reset_index(drop=True)


def load_forecast_2022_data() -> pd.DataFrame:
    """
    Load and validate the 2022 Open-Meteo weather-forecast dataset.
    """
    if not FORECAST_2022_PATH.exists():
        raise FileNotFoundError(
            f"2022 forecast dataset not found: {FORECAST_2022_PATH}"
        )

    df = pd.read_csv(FORECAST_2022_PATH)

    required_columns = {
        "date",
        FEATURE_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"2022 forecast dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df[[FEATURE_COLUMN]].isna().any().any():
        raise ValueError("2022 forecast dataset contains missing values.")

    if df["date"].duplicated().any():
        raise ValueError("2022 forecast dataset contains duplicate dates.")

    return df.sort_values("date").reset_index(drop=True)


def evaluate_model_with_time_split(
    historical_df: pd.DataFrame,
) -> None:
    """
    Evaluate the baseline model using a time-based split.

    Training period: 2017-2020
    Test period:     2021
    """
    training_df = historical_df[
        historical_df["date"].dt.year <= 2020
    ].copy()

    test_df = historical_df[
        historical_df["date"].dt.year == 2021
    ].copy()

    if training_df.empty:
        raise ValueError("Training period 2017-2020 is empty.")

    if test_df.empty:
        raise ValueError("Test period 2021 is empty.")

    x_train = training_df[[FEATURE_COLUMN]]
    y_train = training_df[TARGET_COLUMN]

    x_test = test_df[[FEATURE_COLUMN]]
    y_test = test_df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nBaseline evaluation: train on 2017-2020, test on 2021")
    print("-" * 60)
    print(f"Training rows: {len(training_df)}")
    print(f"Test rows:     {len(test_df)}")
    print(f"MAE:           {mae:,.2f} kWh")
    print(f"RMSE:          {rmse:,.2f} kWh")
    print(f"R² score:      {r2:.4f}")
    print(f"Coefficient:   {model.coef_[0]:,.2f}")
    print(f"Intercept:     {model.intercept_:,.2f}")


def train_final_model_and_predict_2022(
    historical_df: pd.DataFrame,
    forecast_2022_df: pd.DataFrame,
) -> None:
    """
    Train the final baseline model on all historical rows and use the
    archived 2022 weather forecasts to predict daily wind generation.
    """
    x_historical = historical_df[[FEATURE_COLUMN]]
    y_historical = historical_df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(x_historical, y_historical)

    prediction_input = forecast_2022_df[[FEATURE_COLUMN]]

    forecast_2022_df["predicted_wind_generation_kwh"] = model.predict(
        prediction_input
    )

    # Wind generation should not be negative.
    forecast_2022_df["predicted_wind_generation_kwh"] = (
        forecast_2022_df["predicted_wind_generation_kwh"]
        .clip(lower=0)
    )

    output_df = forecast_2022_df[
        [
            "date",
            FEATURE_COLUMN,
            "predicted_wind_generation_kwh",
        ]
    ].copy()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    print("\nFinal baseline model trained on 2017-2021")
    print("-" * 60)
    print(f"Historical rows used: {len(historical_df)}")
    print(f"2022 predictions:     {len(output_df)}")
    print(f"Model saved to:       {MODEL_OUTPUT_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_OUTPUT_PATH}")

    print("\nFirst five 2022 predictions:")
    print(output_df.head())


def main() -> None:
    historical_df = load_historical_data()
    forecast_2022_df = load_forecast_2022_data()

    evaluate_model_with_time_split(historical_df)

    train_final_model_and_predict_2022(
        historical_df,
        forecast_2022_df,
    )


if __name__ == "__main__":
    main()