import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient, UpdateOne

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

PREDICTIONS_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "baseline_linear_regression_predictions_2022.csv"
)

DATABASE_NAME = "wind_energy_project"
COLLECTION_NAME = "baseline_predictions_2022"

load_dotenv(PROJECT_ROOT / ".env")


def load_predictions() -> pd.DataFrame:
    """
    Load and validate the daily baseline predictions.
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
        "predicted_wind_generation_kwh",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.isna().any().any():
        raise ValueError("The predictions dataset contains missing values.")

    if df["date"].duplicated().any():
        raise ValueError("The predictions dataset contains duplicate dates.")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df


def connect_to_collection():
    """
    Connect to MongoDB Atlas and return the predictions collection.
    """
    mongodb_uri = os.getenv("MONGODB_URI")

    if not mongodb_uri:
        raise ValueError(
            "MONGODB_URI was not found. Check your .env file."
        )

    client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )

    client.admin.command("ping")

    database = client[DATABASE_NAME]
    collection = database[COLLECTION_NAME]

    return client, collection


def upload_predictions(
    df: pd.DataFrame,
    collection,
) -> None:
    """
    Insert or update one MongoDB document per day.

    Upsert is used so that the script can safely be rerun later.
    """
    operations = []

    for _, row in df.iterrows():
        document = {
            "date": row["date"],
            "avg_wind_speed_all_stations": float(
                row["avg_wind_speed_all_stations"]
            ),
            "predicted_wind_generation_kwh": float(
                row["predicted_wind_generation_kwh"]
            ),
            "model_name": "baseline_linear_regression",
            "model_version": "v1",
            "prediction_year": 2022,
        }

        operations.append(
            UpdateOne(
                {"date": document["date"]},
                {"$set": document},
                upsert=True,
            )
        )

    result = collection.bulk_write(operations)

    print("MongoDB upload completed successfully.")
    print(f"Inserted documents: {result.upserted_count}")
    print(f"Updated documents:  {result.modified_count}")


def create_date_index(collection) -> None:
    """
    Create a unique index so that each date appears only once.
    """
    index_name = collection.create_index(
        [("date", ASCENDING)],
        unique=True,
    )

    print(f"Created or confirmed index: {index_name}")


def print_summary(collection) -> None:
    """
    Print the collection size and one example document.
    """
    total_documents = collection.count_documents({})

    print(f"Total documents in collection: {total_documents}")

    example_document = collection.find_one(
        {},
        {"_id": 0},
        sort=[("date", ASCENDING)],
    )

    print("\nExample MongoDB document:")
    print(example_document)


def main() -> None:
    predictions_df = load_predictions()

    client, collection = connect_to_collection()

    try:
        create_date_index(collection)
        upload_predictions(predictions_df, collection)
        print_summary(collection)

    finally:
        client.close()


if __name__ == "__main__":
    main()