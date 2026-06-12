from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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

ACTUAL_2022_PATH = (
    SRC_DIR
    / "data"
    / "True_power_generation_in_2022"
    / "wind-2022-uur-data-reformated.csv"
)

OUTPUT_FOLDER = (
    SRC_DIR
    / "data"
    / "model_output"
)

MODEL_OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "random_forest_regression.joblib"
)

PREDICTIONS_OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "random_forest_predictions_2022.csv"
)

COMPARISON_OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "random_forest_comparison_2022.csv"
)

FEATURE_IMPORTANCE_OUTPUT_PATH = (
    OUTPUT_FOLDER
    / "random_forest_feature_importance.csv"
)

TARGET_COLUMN = "daily_wind_generation_kwh"

ACTUAL_2022_TARGET_COLUMN = (
    "Actual Energy created (kWh)"
)

FEATURE_COLUMNS = [
    "avg_wind_speed_all_stations",
    "avg_wind_speed_cubed",
    "wind_direction_sin",
    "wind_direction_cos",
    "installed_wind_capacity_mw",
]


def load_csv(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load a CSV file and verify that it exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found: {path}"
        )

    return pd.read_csv(path)


def validate_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """
    Verify that a dataset contains all required columns.
    """
    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def load_capacity_data() -> pd.DataFrame:
    """
    Load annual installed wind-capacity values.
    """
    capacity_df = load_csv(
        path=CAPACITY_PATH,
        dataset_name="Capacity dataset",
    )

    validate_columns(
        df=capacity_df,
        required_columns={
            "year",
            "installed_wind_capacity_mw",
        },
        dataset_name="Capacity dataset",
    )

    if capacity_df["year"].duplicated().any():
        raise ValueError(
            "Capacity dataset contains duplicate years."
        )

    if (
        capacity_df["installed_wind_capacity_mw"]
        .isna()
        .any()
    ):
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
    Add annual installed capacity to each daily row.
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

    if (
        output_df["installed_wind_capacity_mw"]
        .isna()
        .any()
    ):
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
        "correlation": np.corrcoef(
            actual,
            predicted,
        )[0, 1],
    }


def create_random_forest_model() -> RandomForestRegressor:
    """
    Create a reproducible and intentionally simple
    random-forest regression model.
    """
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


def load_historical_data(
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load historical weather and energy data for 2017-2021.
    """
    historical_df = load_csv(
        path=HISTORICAL_PATH,
        dataset_name="Historical dataset",
    )

    validate_columns(
        df=historical_df,
        required_columns={
            "date",
            TARGET_COLUMN,
            "avg_wind_speed_all_stations",
            "avg_wind_speed_cubed",
            "wind_direction_sin",
            "wind_direction_cos",
        },
        dataset_name="Historical dataset",
    )

    historical_df["date"] = pd.to_datetime(
        historical_df["date"],
        errors="raise",
    )

    if historical_df["date"].duplicated().any():
        raise ValueError(
            "Historical dataset contains duplicate dates."
        )

    historical_df = add_installed_capacity(
        df=historical_df,
        capacity_df=capacity_df,
    )

    if (
        historical_df[
            FEATURE_COLUMNS
            + [TARGET_COLUMN]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Historical dataset contains missing values "
            "in required modelling columns."
        )

    return (
        historical_df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_forecast_2022_data(
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load archived Open-Meteo daily forecasts for 2022.
    """
    forecast_df = load_csv(
        path=FORECAST_2022_PATH,
        dataset_name="2022 Open-Meteo forecast dataset",
    )

    validate_columns(
        df=forecast_df,
        required_columns={
            "date",
            "avg_wind_speed_all_stations",
            "avg_wind_speed_cubed",
            "wind_direction_sin",
            "wind_direction_cos",
        },
        dataset_name="2022 Open-Meteo forecast dataset",
    )

    forecast_df["date"] = pd.to_datetime(
        forecast_df["date"],
        errors="raise",
    )

    if forecast_df["date"].duplicated().any():
        raise ValueError(
            "2022 forecast dataset contains duplicate dates."
        )

    forecast_df = add_installed_capacity(
        df=forecast_df,
        capacity_df=capacity_df,
    )

    if (
        forecast_df[FEATURE_COLUMNS]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "2022 forecast dataset contains missing values "
            "in required modelling columns."
        )

    return (
        forecast_df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_actual_2022_data() -> pd.DataFrame:
    """
    Load actual daily wind-energy generation for 2022.
    """
    actual_df = load_csv(
        path=ACTUAL_2022_PATH,
        dataset_name="Actual 2022 energy dataset",
    )

    validate_columns(
        df=actual_df,
        required_columns={
            "date",
            ACTUAL_2022_TARGET_COLUMN,
        },
        dataset_name="Actual 2022 energy dataset",
    )

    actual_df["date"] = pd.to_datetime(
        actual_df["date"],
        errors="raise",
    )

    if actual_df["date"].duplicated().any():
        raise ValueError(
            "Actual 2022 dataset contains duplicate dates."
        )

    return (
        actual_df[
            [
                "date",
                ACTUAL_2022_TARGET_COLUMN,
            ]
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )


def print_metrics(
    heading: str,
    metrics: dict[str, float],
) -> None:
    """
    Print regression metrics in a readable format.
    """
    print(f"\n{heading}")
    print("-" * 72)

    print(
        f"MAE:         {metrics['mae']:,.2f} kWh"
    )

    print(
        f"RMSE:        {metrics['rmse']:,.2f} kWh"
    )

    print(
        f"R²:          {metrics['r2']:.4f}"
    )

    print(
        f"Correlation: {metrics['correlation']:.4f}"
    )


def evaluate_validation_period(
    historical_df: pd.DataFrame,
) -> None:
    """
    Train on 2017-2020 and validate on 2021.
    """
    training_df = historical_df[
        historical_df["date"].dt.year <= 2020
    ].copy()

    validation_df = historical_df[
        historical_df["date"].dt.year == 2021
    ].copy()

    if training_df.empty:
        raise ValueError(
            "No training rows found for 2017-2020."
        )

    if validation_df.empty:
        raise ValueError(
            "No validation rows found for 2021."
        )

    model = create_random_forest_model()

    model.fit(
        training_df[FEATURE_COLUMNS],
        training_df[TARGET_COLUMN],
    )

    predictions = model.predict(
        validation_df[FEATURE_COLUMNS]
    )

    metrics = calculate_metrics(
        actual=validation_df[TARGET_COLUMN],
        predicted=predictions,
    )

    print_metrics(
        heading=(
            "Random-forest validation: "
            "train on 2017-2020, validate on 2021"
        ),
        metrics=metrics,
    )


def train_final_model_and_evaluate_2022(
    historical_df: pd.DataFrame,
    forecast_2022_df: pd.DataFrame,
    actual_2022_df: pd.DataFrame,
) -> None:
    """
    Train on 2017-2021, predict 2022 and compare with
    actual 2022 generation.
    """
    model = create_random_forest_model()

    model.fit(
        historical_df[FEATURE_COLUMNS],
        historical_df[TARGET_COLUMN],
    )

    predictions_df = forecast_2022_df.copy()

    predictions_df[
        "predicted_wind_generation_kwh"
    ] = model.predict(
        predictions_df[FEATURE_COLUMNS]
    )

    predictions_df[
        "predicted_wind_generation_kwh"
    ] = (
        predictions_df[
            "predicted_wind_generation_kwh"
        ]
        .clip(lower=0)
    )

    prediction_output_columns = [
        "date",
        *FEATURE_COLUMNS,
        "predicted_wind_generation_kwh",
    ]

    predictions_df = predictions_df[
        prediction_output_columns
    ]

    comparison_df = predictions_df.merge(
        actual_2022_df,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    if comparison_df.empty:
        raise ValueError(
            "No matching dates found for the "
            "2022 comparison."
        )

    comparison_df["error_kwh"] = (
        comparison_df[
            "predicted_wind_generation_kwh"
        ]
        - comparison_df[
            ACTUAL_2022_TARGET_COLUMN
        ]
    )

    comparison_df["absolute_error_kwh"] = (
        comparison_df["error_kwh"].abs()
    )

    comparison_df["percentage_error"] = np.where(
        comparison_df[
            ACTUAL_2022_TARGET_COLUMN
        ] != 0,
        (
            comparison_df["absolute_error_kwh"]
            / comparison_df[
                ACTUAL_2022_TARGET_COLUMN
            ]
        )
        * 100,
        np.nan,
    )

    metrics = calculate_metrics(
        actual=comparison_df[
            ACTUAL_2022_TARGET_COLUMN
        ],
        predicted=comparison_df[
            "predicted_wind_generation_kwh"
        ],
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    comparison_df.to_csv(
        COMPARISON_OUTPUT_PATH,
        index=False,
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_OUTPUT_PATH,
        index=False,
    )

    print_metrics(
        heading=(
            "Random-forest external evaluation: "
            "train on 2017-2021, evaluate on 2022"
        ),
        metrics=metrics,
    )

    print(
        "\nPrediction level comparison"
    )

    print("-" * 72)

    print(
        "Average actual generation:    "
        f"{comparison_df[ACTUAL_2022_TARGET_COLUMN].mean():,.2f} kWh"
    )

    print(
        "Average predicted generation: "
        f"{comparison_df['predicted_wind_generation_kwh'].mean():,.2f} kWh"
    )

    print(
        "Average prediction bias:      "
        f"{comparison_df['error_kwh'].mean():,.2f} kWh"
    )

    print(
        "\nPrediction direction counts"
    )

    print("-" * 72)

    print(
        "Underestimations: "
        f"{(comparison_df['error_kwh'] < 0).sum()}"
    )

    print(
        "Overestimations:  "
        f"{(comparison_df['error_kwh'] > 0).sum()}"
    )

    print(
        "Zero predictions: "
        f"{(comparison_df['predicted_wind_generation_kwh'] == 0).sum()}"
    )

    print(
        "\nAverage absolute percentage error: "
        f"{comparison_df['percentage_error'].mean():.2f}%"
    )

    print(
        "\nFeature importance"
    )

    print("-" * 72)

    print(
        feature_importance_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\nLargest five prediction errors"
    )

    print("-" * 72)

    print(
        comparison_df[
            [
                "date",
                "predicted_wind_generation_kwh",
                ACTUAL_2022_TARGET_COLUMN,
                "absolute_error_kwh",
            ]
        ]
        .sort_values(
            "absolute_error_kwh",
            ascending=False,
        )
        .head()
        .to_string(index=False)
    )

    print(
        "\nSaved files"
    )

    print("-" * 72)

    print(
        f"Model:              {MODEL_OUTPUT_PATH}"
    )

    print(
        f"Predictions:        {PREDICTIONS_OUTPUT_PATH}"
    )

    print(
        f"2022 comparison:    {COMPARISON_OUTPUT_PATH}"
    )

    print(
        f"Feature importance: "
        f"{FEATURE_IMPORTANCE_OUTPUT_PATH}"
    )


def main() -> None:
    capacity_df = load_capacity_data()

    historical_df = load_historical_data(
        capacity_df=capacity_df,
    )

    forecast_2022_df = load_forecast_2022_data(
        capacity_df=capacity_df,
    )

    actual_2022_df = load_actual_2022_data()

    evaluate_validation_period(
        historical_df=historical_df,
    )

    train_final_model_and_evaluate_2022(
        historical_df=historical_df,
        forecast_2022_df=forecast_2022_df,
        actual_2022_df=actual_2022_df,
    )


if __name__ == "__main__":
    main()