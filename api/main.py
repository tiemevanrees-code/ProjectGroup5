import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pymongo import ASCENDING, MongoClient

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError(
        "MONGODB_URI was not found. Check your .env file."
    )

DATABASE_NAME = "wind_energy_project"
COLLECTION_NAME = "baseline_predictions_2022"

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
)

database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

app = FastAPI(
    title="Wind Energy Prediction API",
    description=(
        "API for retrieving predicted daily wind-energy generation "
        "based on archived 2022 weather forecasts."
    ),
    version="1.0.0",
)


def validate_date(date: str) -> str:
    """
    Require dates in YYYY-MM-DD format.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD.",
        ) from error

    return date


@app.get("/")
def root() -> dict:
    """
    Basic API status endpoint.
    """
    return {
        "message": "Wind Energy Prediction API is running.",
        "model": "baseline_linear_regression",
        "prediction_year": 2022,
    }


@app.get("/predictions/{date}")
def get_prediction_by_date(date: str) -> dict:
    """
    Retrieve one wind-energy prediction for a specific date.

    Example:
    GET /predictions/2022-01-15
    """
    validated_date = validate_date(date)

    document = collection.find_one(
        {"date": validated_date},
        {"_id": 0},
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for {validated_date}.",
        )

    return document


@app.get("/predictions")
def get_predictions(
    limit: int = Query(default=30, ge=1, le=365),
) -> list[dict]:
    """
    Retrieve multiple predictions in chronological order.

    Example:
    GET /predictions?limit=10
    """
    documents = (
        collection
        .find({}, {"_id": 0})
        .sort("date", ASCENDING)
        .limit(limit)
    )

    return list(documents)