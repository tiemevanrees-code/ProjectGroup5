import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pymongo import ASCENDING, MongoClient

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)

MONGODB_URI = os.getenv(
    "MONGODB_URI"
)

if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI was not found. "
        "Check your .env file."
    )

DATABASE_NAME = "wind_energy_project"

COLLECTION_NAME = (
    "daily_prediction_results_2022"
)

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
)

database = client[
    DATABASE_NAME
]

collection = database[
    COLLECTION_NAME
]

app = FastAPI(
    title="Wind Energy Prediction API",
    description=(
        "API for retrieving daily Dutch wind-energy predictions, "
        "actual generation values, weather features, and "
        "model-evaluation insights for 2022."
    ),
    version="2.0.0",
)


def validate_date(
    date_value: str,
) -> str:
    """
    Require a date in YYYY-MM-DD format.
    """
    try:
        datetime.strptime(
            date_value,
            "%Y-%m-%d",
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid date format. "
                "Use YYYY-MM-DD."
            ),
        ) from error

    return date_value


def calculate_r2(
    actual_values: list[float],
    predicted_values: list[float],
) -> float:
    """
    Calculate the R² score.
    """
    if not actual_values:
        raise ValueError(
            "Cannot calculate R² without values."
        )

    average_actual = (
        sum(actual_values)
        / len(actual_values)
    )

    sum_squared_residuals = sum(
        (
            actual
            - predicted
        ) ** 2
        for actual, predicted in zip(
            actual_values,
            predicted_values,
        )
    )

    sum_squared_total = sum(
        (
            actual
            - average_actual
        ) ** 2
        for actual in actual_values
    )

    if sum_squared_total == 0:
        return 0.0

    return (
        1
        - (
            sum_squared_residuals
            / sum_squared_total
        )
    )


def calculate_correlation(
    actual_values: list[float],
    predicted_values: list[float],
) -> float:
    """
    Calculate the Pearson correlation coefficient.
    """
    if not actual_values:
        raise ValueError(
            "Cannot calculate correlation without values."
        )

    average_actual = (
        sum(actual_values)
        / len(actual_values)
    )

    average_predicted = (
        sum(predicted_values)
        / len(predicted_values)
    )

    numerator = sum(
        (
            actual
            - average_actual
        )
        * (
            predicted
            - average_predicted
        )
        for actual, predicted in zip(
            actual_values,
            predicted_values,
        )
    )

    denominator_actual = math.sqrt(
        sum(
            (
                actual
                - average_actual
            ) ** 2
            for actual in actual_values
        )
    )

    denominator_predicted = math.sqrt(
        sum(
            (
                predicted
                - average_predicted
            ) ** 2
            for predicted in predicted_values
        )
    )

    denominator = (
        denominator_actual
        * denominator_predicted
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_date_filter(
    start_date: str | None,
    end_date: str | None,
) -> dict[str, Any]:
    """
    Build an optional MongoDB filter for a date range.
    """
    date_filter: dict[str, str] = {}

    if start_date is not None:
        date_filter["$gte"] = validate_date(
            start_date
        )

    if end_date is not None:
        date_filter["$lte"] = validate_date(
            end_date
        )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "start_date cannot be later "
                "than end_date."
            ),
        )

    if not date_filter:
        return {}

    return {
        "date": date_filter
    }


@app.get("/")
def root() -> dict:
    """
    Basic API status endpoint.
    """
    return {
        "message": (
            "Wind Energy Prediction API "
            "is running."
        ),
        "model_name": (
            "multivariable_linear_regression"
        ),
        "model_version": (
            "v1_capacity_adjusted"
        ),
        "prediction_year": 2022,
        "available_endpoints": [
            "/predictions",
            "/predictions/{date}",
            "/summary",
            "/docs",
        ],
    }


@app.get("/predictions/{date}")
def get_prediction_by_date(
    date: str,
) -> dict:
    """
    Retrieve one daily prediction result.

    Example:
    GET /predictions/2022-01-15
    """
    validated_date = validate_date(
        date
    )

    document = collection.find_one(
        {
            "date": validated_date
        },
        {
            "_id": 0
        },
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No prediction result found "
                f"for {validated_date}."
            ),
        )

    return document


@app.get("/predictions")
def get_predictions(
    start_date: str | None = Query(
        default=None,
        description=(
            "Optional start date in "
            "YYYY-MM-DD format."
        ),
    ),
    end_date: str | None = Query(
        default=None,
        description=(
            "Optional end date in "
            "YYYY-MM-DD format."
        ),
    ),
    limit: int = Query(
        default=30,
        ge=1,
        le=365,
        description=(
            "Maximum number of results."
        ),
    ),
) -> list[dict]:
    """
    Retrieve daily prediction results in chronological order.

    Examples:
    GET /predictions
    GET /predictions?limit=10
    GET /predictions?start_date=2022-01-01&end_date=2022-01-31
    """
    query_filter = build_date_filter(
        start_date=start_date,
        end_date=end_date,
    )

    documents = (
        collection
        .find(
            query_filter,
            {
                "_id": 0
            },
        )
        .sort(
            "date",
            ASCENDING,
        )
        .limit(
            limit
        )
    )

    return list(
        documents
    )


@app.get("/summary")
def get_summary(
    start_date: str | None = Query(
        default=None,
        description=(
            "Optional start date in "
            "YYYY-MM-DD format."
        ),
    ),
    end_date: str | None = Query(
        default=None,
        description=(
            "Optional end date in "
            "YYYY-MM-DD format."
        ),
    ),
) -> dict:
    """
    Return aggregated model-performance insights.

    Examples:
    GET /summary
    GET /summary?start_date=2022-01-01&end_date=2022-03-31
    """
    query_filter = build_date_filter(
        start_date=start_date,
        end_date=end_date,
    )

    documents = list(
        collection.find(
            query_filter,
            {
                "_id": 0,
                "date": 1,
                "prediction": 1,
            },
        )
        .sort(
            "date",
            ASCENDING,
        )
    )

    if not documents:
        raise HTTPException(
            status_code=404,
            detail=(
                "No prediction results found "
                "for the selected date range."
            ),
        )

    actual_values = [
        float(
            document[
                "prediction"
            ][
                "actual_wind_generation_kwh"
            ]
        )
        for document in documents
    ]

    predicted_values = [
        float(
            document[
                "prediction"
            ][
                "predicted_wind_generation_kwh"
            ]
        )
        for document in documents
    ]

    errors = [
        predicted
        - actual
        for actual, predicted in zip(
            actual_values,
            predicted_values,
        )
    ]

    absolute_errors = [
        abs(
            error
        )
        for error in errors
    ]

    percentage_errors = [
        (
            absolute_error
            / actual
        )
        * 100
        for actual, absolute_error in zip(
            actual_values,
            absolute_errors,
        )
        if actual != 0
    ]

    mean_absolute_error = (
        sum(
            absolute_errors
        )
        / len(
            absolute_errors
        )
    )

    root_mean_squared_error = math.sqrt(
        sum(
            error ** 2
            for error in errors
        )
        / len(
            errors
        )
    )

    largest_error_document = max(
        documents,
        key=lambda document: (
            document[
                "prediction"
            ][
                "absolute_error_kwh"
            ]
        ),
    )

    return {
        "model_name": (
            "multivariable_linear_regression"
        ),
        "model_version": (
            "v1_capacity_adjusted"
        ),
        "number_of_days": len(
            documents
        ),
        "date_range": {
            "start_date": documents[
                0
            ][
                "date"
            ],
            "end_date": documents[
                -1
            ][
                "date"
            ],
        },
        "average_actual_generation_kwh": (
            sum(
                actual_values
            )
            / len(
                actual_values
            )
        ),
        "average_predicted_generation_kwh": (
            sum(
                predicted_values
            )
            / len(
                predicted_values
            )
        ),
        "average_prediction_bias_kwh": (
            sum(
                errors
            )
            / len(
                errors
            )
        ),
        "mae_kwh": (
            mean_absolute_error
        ),
        "rmse_kwh": (
            root_mean_squared_error
        ),
        "r2": calculate_r2(
            actual_values=actual_values,
            predicted_values=predicted_values,
        ),
        "correlation": calculate_correlation(
            actual_values=actual_values,
            predicted_values=predicted_values,
        ),
        "average_percentage_error": (
            sum(
                percentage_errors
            )
            / len(
                percentage_errors
            )
            if percentage_errors
            else None
        ),
        "underestimations": sum(
            error < 0
            for error in errors
        ),
        "overestimations": sum(
            error > 0
            for error in errors
        ),
        "largest_absolute_error": {
            "date": (
                largest_error_document[
                    "date"
                ]
            ),
            "absolute_error_kwh": (
                largest_error_document[
                    "prediction"
                ][
                    "absolute_error_kwh"
                ]
            ),
        },
    }