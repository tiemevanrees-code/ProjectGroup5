from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "multivariable_linear_regression_predictions_2022.csv"
)

ACTUAL_VALUES_PATH = (
    SRC_DIR
    / "data"
    / "True_power_generation_in_2022"
    / "wind-2022-uur-data-reformated.csv"
)

OUTPUT_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "enriched_prediction_results_2022.csv"
)

PREDICTION_COLUMN = (
    "predicted_wind_generation_kwh"
)

ACTUAL_COLUMN = (
    "Actual Energy created (kWh)"
)

MODEL_NAME = (
    "multivariable_linear_regression"
)

MODEL_VERSION = (
    "v1_capacity_adjusted"
)


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
    Verify that all required columns are present.
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


def main() -> None:
    predictions_df = load_csv(
        path=PREDICTIONS_PATH,
        dataset_name="Selected-model prediction dataset",
    )

    actual_df = load_csv(
        path=ACTUAL_VALUES_PATH,
        dataset_name="Actual 2022 wind-generation dataset",
    )

    validate_columns(
        df=predictions_df,
        required_columns={
            "date",
            "avg_wind_speed_all_stations",
            "avg_wind_speed_cubed",
            "wind_direction_sin",
            "wind_direction_cos",
            "installed_wind_capacity_mw",
            PREDICTION_COLUMN,
        },
        dataset_name="Selected-model prediction dataset",
    )

    validate_columns(
        df=actual_df,
        required_columns={
            "date",
            ACTUAL_COLUMN,
        },
        dataset_name="Actual 2022 wind-generation dataset",
    )

    predictions_df["date"] = pd.to_datetime(
        predictions_df["date"],
        errors="raise",
    )

    actual_df["date"] = pd.to_datetime(
        actual_df["date"],
        errors="raise",
    )

    if predictions_df["date"].duplicated().any():
        raise ValueError(
            "Prediction dataset contains duplicate dates."
        )

    if actual_df["date"].duplicated().any():
        raise ValueError(
            "Actual dataset contains duplicate dates."
        )

    enriched_df = predictions_df.merge(
        actual_df[
            [
                "date",
                ACTUAL_COLUMN,
            ]
        ],
        on="date",
        how="inner",
        validate="one_to_one",
    )

    if enriched_df.empty:
        raise ValueError(
            "No matching dates were found between "
            "the prediction and actual datasets."
        )

    enriched_df = enriched_df.rename(
        columns={
            ACTUAL_COLUMN:
                "actual_wind_generation_kwh"
        }
    )

    enriched_df["error_kwh"] = (
        enriched_df[
            "predicted_wind_generation_kwh"
        ]
        - enriched_df[
            "actual_wind_generation_kwh"
        ]
    )

    enriched_df["absolute_error_kwh"] = (
        enriched_df["error_kwh"].abs()
    )

    enriched_df["percentage_error"] = np.where(
        enriched_df[
            "actual_wind_generation_kwh"
        ] != 0,
        (
            enriched_df["absolute_error_kwh"]
            / enriched_df[
                "actual_wind_generation_kwh"
            ]
        )
        * 100,
        np.nan,
    )

    enriched_df["model_name"] = MODEL_NAME

    enriched_df["model_version"] = MODEL_VERSION

    enriched_df["date"] = (
        enriched_df["date"]
        .dt.strftime("%Y-%m-%d")
    )

    output_columns = [
        "date",
        "model_name",
        "model_version",
        "avg_wind_speed_all_stations",
        "avg_wind_speed_cubed",
        "wind_direction_sin",
        "wind_direction_cos",
        "installed_wind_capacity_mw",
        "predicted_wind_generation_kwh",
        "actual_wind_generation_kwh",
        "error_kwh",
        "absolute_error_kwh",
        "percentage_error",
    ]

    enriched_df = enriched_df[
        output_columns
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nEnriched 2022 prediction results created"
    )

    print("-" * 72)

    print(
        f"Rows saved: {len(enriched_df)}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    print(
        "\nFirst five rows:"
    )

    print(
        enriched_df
        .head()
        .to_string(index=False)
    )

    print(
        "\nSummary"
    )

    print("-" * 72)

    print(
        "Average actual generation:    "
        f"{enriched_df['actual_wind_generation_kwh'].mean():,.2f} kWh"
    )

    print(
        "Average predicted generation: "
        f"{enriched_df['predicted_wind_generation_kwh'].mean():,.2f} kWh"
    )

    print(
        "Average absolute error:       "
        f"{enriched_df['absolute_error_kwh'].mean():,.2f} kWh"
    )

    print(
        "Average percentage error:     "
        f"{enriched_df['percentage_error'].mean():.2f}%"
    )


if __name__ == "__main__":
    main()