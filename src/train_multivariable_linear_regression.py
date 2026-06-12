from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

HISTORICAL_PATH = (
    PROJECT_ROOT
    / "MVP"
    / "merged_weather_energy_daily_mvp.csv"
)

FORECAST_2022_PATH = (
    SRC_DIR
    / "data"
    / "processed"
    / "open_meteo_forecast_2022_daily.csv"
)

CAPACITY_PATH = (
    SRC_DIR
    / "data"
    / "raw"
    / "wind_capacity_by_year.csv"
)

OUTPUT_FOLDER = (
    SRC_DIR
    / "data"
    / "model_output"
)

PREDICTIONS_OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "multivariable_linear_regression_predictions_2022.csv"
)

MODEL_OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "multivariable_linear_regression.joblib"
)

BASE_WEATHER_FEATURE_COLUMNS = [
    "avg_wind_speed_all_stations",
    "avg_wind_speed_cubed",
    "wind_direction_sin",
    "wind_direction_cos",
]

FEATURE_COLUMNS = [
    *BASE_WEATHER_FEATURE_COLUMNS,
    "installed_wind_capacity_mw",
]

TARGET_COLUMN = "daily_wind_generation_kwh"


def load_and_validate_csv(
    path: Path,
    required_columns: set[str],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load a CSV file and perform basic validation.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    columns_to_check = list(
        required_columns - {"date"}
    )

    if df[columns_to_check].isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing values "
            f"in required columns."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            f"{dataset_name} contains duplicate dates."
        )

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_historical_data() -> pd.DataFrame:
    """
    Load historical weather and wind-generation data for 2017-2021.
    """
    return load_and_validate_csv(
        path=HISTORICAL_PATH,
        required_columns={
            "date",
            TARGET_COLUMN,
            *BASE_WEATHER_FEATURE_COLUMNS,
        },
        dataset_name="Historical dataset",
    )


def load_forecast_2022_data() -> pd.DataFrame:
    """
    Load archived Open-Meteo forecast data for 2022.
    """
    return load_and_validate_csv(
        path=FORECAST_2022_PATH,
        required_columns={
            "date",
            *BASE_WEATHER_FEATURE_COLUMNS,
        },
        dataset_name="2022 Open-Meteo forecast dataset",
    )


def load_capacity_data() -> pd.DataFrame:
    """
    Load annual Dutch installed wind capacity.
    """
    if not CAPACITY_PATH.exists():
        raise FileNotFoundError(
            f"Capacity dataset not found: {CAPACITY_PATH}"
        )

    capacity_df = pd.read_csv(CAPACITY_PATH)

    required_columns = {
        "year",
        "installed_wind_capacity_mw",
    }

    missing_columns = required_columns - set(
        capacity_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Capacity dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if capacity_df["year"].duplicated().any():
        raise ValueError(
            "Capacity dataset contains duplicate years."
        )

    if capacity_df[
        "installed_wind_capacity_mw"
    ].isna().any():
        raise ValueError(
            "Capacity dataset contains missing values."
        )

    if (
        capacity_df["installed_wind_capacity_mw"]
        <= 0
    ).any():
        raise ValueError(
            "Installed-capacity values must be positive."
        )

    return capacity_df


def add_installed_capacity(
    df: pd.DataFrame,
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the annual installed-capacity value to each daily row.
    """
    output_df = df.copy()

    output_df["year"] = (
        output_df["date"].dt.year
    )

    output_df = output_df.merge(
        capacity_df,
        on="year",
        how="left",
        validate="many_to_one",
    )

    if output_df[
        "installed_wind_capacity_mw"
    ].isna().any():
        missing_years = sorted(
            output_df.loc[
                output_df[
                    "installed_wind_capacity_mw"
                ].isna(),
                "year",
            ].unique()
        )

        raise ValueError(
            "Missing installed-capacity values "
            f"for years: {missing_years}"
        )

    return output_df


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression metrics.
    """
    return {
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "r2": r2_score(
            actual,
            predicted,
        ),
    }


def evaluate_models_with_time_split(
    historical_df: pd.DataFrame,
) -> None:
    """
    Compare the original one-input baseline model with the
    improved multivariable model.

    Training period: 2017-2020
    Validation period: 2021
    """
    training_df = historical_df[
        historical_df["date"].dt.year <= 2020
    ].copy()

    validation_df = historical_df[
        historical_df["date"].dt.year == 2021
    ].copy()

    if training_df.empty:
        raise ValueError(
            "No training data found for 2017-2020."
        )

    if validation_df.empty:
        raise ValueError(
            "No validation data found for 2021."
        )

    y_train = training_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]

    baseline_model = LinearRegression()

    baseline_model.fit(
        training_df[
            ["avg_wind_speed_all_stations"]
        ],
        y_train,
    )

    baseline_predictions = baseline_model.predict(
        validation_df[
            ["avg_wind_speed_all_stations"]
        ]
    )

    baseline_metrics = calculate_metrics(
        actual=y_validation,
        predicted=baseline_predictions,
    )

    multivariable_model = LinearRegression()

    multivariable_model.fit(
        training_df[FEATURE_COLUMNS],
        y_train,
    )

    multivariable_predictions = (
        multivariable_model.predict(
            validation_df[FEATURE_COLUMNS]
        )
    )

    multivariable_metrics = calculate_metrics(
        actual=y_validation,
        predicted=multivariable_predictions,
    )

    print(
        "\nModel comparison: "
        "train on 2017-2020, validate on 2021"
    )

    print("-" * 82)

    print(
        f"{'Model':<40}"
        f"{'MAE (kWh)':>16}"
        f"{'RMSE (kWh)':>16}"
        f"{'R²':>10}"
    )

    print("-" * 82)

    print(
        f"{'Baseline linear regression':<40}"
        f"{baseline_metrics['mae']:>16,.2f}"
        f"{baseline_metrics['rmse']:>16,.2f}"
        f"{baseline_metrics['r2']:>10.4f}"
    )

    print(
        f"{'Multivariable linear regression':<40}"
        f"{multivariable_metrics['mae']:>16,.2f}"
        f"{multivariable_metrics['rmse']:>16,.2f}"
        f"{multivariable_metrics['r2']:>10.4f}"
    )

    print(
        "\nMultivariable model coefficients:"
    )

    print("-" * 82)

    for feature, coefficient in zip(
        FEATURE_COLUMNS,
        multivariable_model.coef_,
    ):
        print(
            f"{feature:<45}"
            f"{coefficient:>20,.2f}"
        )

    print(
        f"{'intercept':<45}"
        f"{multivariable_model.intercept_:>20,.2f}"
    )


def train_final_model_and_predict_2022(
    historical_df: pd.DataFrame,
    forecast_2022_df: pd.DataFrame,
) -> None:
    """
    Train the final multivariable model on 2017-2021 and
    create daily 2022 predictions.
    """
    model = LinearRegression()

    model.fit(
        historical_df[FEATURE_COLUMNS],
        historical_df[TARGET_COLUMN],
    )

    output_df = forecast_2022_df.copy()

    output_df[
        "predicted_wind_generation_kwh"
    ] = model.predict(
        output_df[FEATURE_COLUMNS]
    )

    output_df[
        "predicted_wind_generation_kwh"
    ] = (
        output_df[
            "predicted_wind_generation_kwh"
        ]
        .clip(lower=0)
    )

    output_columns = [
        "date",
        *BASE_WEATHER_FEATURE_COLUMNS,
        "installed_wind_capacity_mw",
        "predicted_wind_generation_kwh",
    ]

    output_df = output_df[output_columns]

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    print(
        "\nFinal multivariable model "
        "trained on 2017-2021"
    )

    print("-" * 82)

    print(
        f"Historical rows used: {len(historical_df)}"
    )

    print(
        f"2022 predictions:     {len(output_df)}"
    )

    print(
        f"Model saved to:       {MODEL_OUTPUT_PATH}"
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_OUTPUT_PATH}"
    )

    print(
        "\nFirst five 2022 predictions:"
    )

    print(
        output_df
        .head()
        .to_string(index=False)
    )


def main() -> None:
    historical_df = load_historical_data()

    forecast_2022_df = load_forecast_2022_data()

    capacity_df = load_capacity_data()

    historical_df = add_installed_capacity(
        historical_df,
        capacity_df,
    )

    forecast_2022_df = add_installed_capacity(
        forecast_2022_df,
        capacity_df,
    )

    evaluate_models_with_time_split(
        historical_df,
    )

    train_final_model_and_predict_2022(
        historical_df,
        forecast_2022_df,
    )


if __name__ == "__main__":
    main()