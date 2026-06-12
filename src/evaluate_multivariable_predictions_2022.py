from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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
    / "multivariable_linear_regression_comparison_2022.csv"
)

PREDICTION_COLUMN = (
    "predicted_wind_generation_kwh"
)

ACTUAL_COLUMN = (
    "Actual Energy created (kWh)"
)


def load_csv(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load a CSV file and check whether it exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} not found: {path}"
        )

    return pd.read_csv(path)


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
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
        "correlation": actual.corr(
            predicted
        ),
    }


def main() -> None:
    predictions_df = load_csv(
        path=PREDICTIONS_PATH,
        dataset_name="Prediction dataset",
    )

    actual_df = load_csv(
        path=ACTUAL_VALUES_PATH,
        dataset_name="Actual 2022 energy dataset",
    )

    required_prediction_columns = {
        "date",
        PREDICTION_COLUMN,
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(predictions_df.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction dataset is missing columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    required_actual_columns = {
        "date",
        ACTUAL_COLUMN,
    }

    missing_actual_columns = (
        required_actual_columns
        - set(actual_df.columns)
    )

    if missing_actual_columns:
        raise ValueError(
            "Actual 2022 dataset is missing columns: "
            f"{sorted(missing_actual_columns)}"
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

    comparison_df = predictions_df.merge(
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

    if comparison_df.empty:
        raise ValueError(
            "No matching dates were found between "
            "the prediction and actual-value datasets."
        )

    comparison_df["error_kwh"] = (
        comparison_df[PREDICTION_COLUMN]
        - comparison_df[ACTUAL_COLUMN]
    )

    comparison_df["absolute_error_kwh"] = (
        comparison_df["error_kwh"].abs()
    )

    comparison_df["percentage_error"] = np.where(
        comparison_df[ACTUAL_COLUMN] != 0,
        (
            comparison_df["absolute_error_kwh"]
            / comparison_df[ACTUAL_COLUMN]
        )
        * 100,
        np.nan,
    )

    metrics = calculate_metrics(
        actual=comparison_df[ACTUAL_COLUMN],
        predicted=comparison_df[
            PREDICTION_COLUMN
        ],
    )

    underestimation_count = (
        comparison_df["error_kwh"] < 0
    ).sum()

    overestimation_count = (
        comparison_df["error_kwh"] > 0
    ).sum()

    exact_match_count = (
        comparison_df["error_kwh"] == 0
    ).sum()

    zero_prediction_count = (
        comparison_df[PREDICTION_COLUMN] == 0
    ).sum()

    average_actual = comparison_df[
        ACTUAL_COLUMN
    ].mean()

    average_predicted = comparison_df[
        PREDICTION_COLUMN
    ].mean()

    average_bias = comparison_df[
        "error_kwh"
    ].mean()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n2022 multivariable "
        "linear-regression evaluation"
    )

    print("-" * 72)

    print(
        f"Matched daily rows: {len(comparison_df)}"
    )

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

    print(
        "\nPrediction level comparison"
    )

    print("-" * 72)

    print(
        f"Average actual generation:    "
        f"{average_actual:,.2f} kWh"
    )

    print(
        f"Average predicted generation: "
        f"{average_predicted:,.2f} kWh"
    )

    print(
        f"Average prediction bias:      "
        f"{average_bias:,.2f} kWh"
    )

    print(
        "\nPrediction direction counts"
    )

    print("-" * 72)

    print(
        f"Underestimations: {underestimation_count}"
    )

    print(
        f"Overestimations:  {overestimation_count}"
    )

    print(
        f"Exact matches:    {exact_match_count}"
    )

    print(
        f"Zero predictions: {zero_prediction_count}"
    )

    print(
        "\nAverage absolute percentage error: "
        f"{comparison_df['percentage_error'].mean():.2f}%"
    )

    print(
        "\nLargest five prediction errors"
    )

    print("-" * 72)

    largest_errors = (
        comparison_df[
            [
                "date",
                PREDICTION_COLUMN,
                ACTUAL_COLUMN,
                "absolute_error_kwh",
            ]
        ]
        .sort_values(
            "absolute_error_kwh",
            ascending=False,
        )
        .head()
    )

    print(
        largest_errors.to_string(
            index=False,
        )
    )

    print(
        f"\nComparison file saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()