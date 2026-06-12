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
    / "enriched_prediction_results_2022.csv"
)

DATABASE_NAME = "wind_energy_project"

COLLECTION_NAME = "daily_prediction_results_2022"

load_dotenv(
    PROJECT_ROOT / ".env"
)


def load_prediction_results() -> pd.DataFrame:
    """
    Load and validate the enriched daily prediction results.
    """
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction-results file not found: "
            f"{PREDICTIONS_PATH}\n"
            "Run create_enriched_prediction_results.py first."
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    required_columns = {
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
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction-results dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.isna().any().any():
        raise ValueError(
            "Prediction-results dataset contains missing values."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Prediction-results dataset contains duplicate dates."
        )

    df["date"] = (
        pd.to_datetime(
            df["date"],
            errors="raise",
        )
        .dt.strftime("%Y-%m-%d")
    )

    return df


def connect_to_collection():
    """
    Connect to MongoDB Atlas and return the selected collection.
    """
    mongodb_uri = os.getenv(
        "MONGODB_URI"
    )

    if not mongodb_uri:
        raise ValueError(
            "MONGODB_URI was not found. "
            "Check your .env file."
        )

    client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=5000,
    )

    client.admin.command(
        "ping"
    )

    database = client[
        DATABASE_NAME
    ]

    collection = database[
        COLLECTION_NAME
    ]

    return client, collection


def create_date_index(
    collection,
) -> None:
    """
    Create a unique ascending index on date.

    This prevents duplicate daily records and speeds up
    API queries for a specific date or date range.
    """
    index_name = (
        collection.create_index(
            [
                (
                    "date",
                    ASCENDING,
                )
            ],
            unique=True,
        )
    )

    print(
        f"Created or confirmed index: "
        f"{index_name}"
    )


def upload_prediction_results(
    df: pd.DataFrame,
    collection,
) -> None:
    """
    Insert or update one MongoDB document per day.

    Upsert is used so that the script can safely be rerun.
    """
    operations = []

    for _, row in df.iterrows():
        document = {
            "date": row["date"],
            "model_name": row[
                "model_name"
            ],
            "model_version": row[
                "model_version"
            ],
            "prediction_year": 2022,
            "weather": {
                "avg_wind_speed_all_stations": float(
                    row[
                        "avg_wind_speed_all_stations"
                    ]
                ),
                "avg_wind_speed_cubed": float(
                    row[
                        "avg_wind_speed_cubed"
                    ]
                ),
                "wind_direction_sin": float(
                    row[
                        "wind_direction_sin"
                    ]
                ),
                "wind_direction_cos": float(
                    row[
                        "wind_direction_cos"
                    ]
                ),
            },
            "installed_wind_capacity_mw": int(
                row[
                    "installed_wind_capacity_mw"
                ]
            ),
            "prediction": {
                "predicted_wind_generation_kwh": float(
                    row[
                        "predicted_wind_generation_kwh"
                    ]
                ),
                "actual_wind_generation_kwh": float(
                    row[
                        "actual_wind_generation_kwh"
                    ]
                ),
                "error_kwh": float(
                    row[
                        "error_kwh"
                    ]
                ),
                "absolute_error_kwh": float(
                    row[
                        "absolute_error_kwh"
                    ]
                ),
                "percentage_error": float(
                    row[
                        "percentage_error"
                    ]
                ),
            },
        }

        operations.append(
            UpdateOne(
                {
                    "date": document[
                        "date"
                    ]
                },
                {
                    "$set": document
                },
                upsert=True,
            )
        )

    result = collection.bulk_write(
        operations
    )

    print(
        "\nMongoDB upload completed successfully."
    )

    print(
        f"Inserted documents: "
        f"{result.upserted_count}"
    )

    print(
        f"Updated documents:  "
        f"{result.modified_count}"
    )


def print_summary(
    collection,
) -> None:
    """
    Print the number of documents and one example document.
    """
    total_documents = (
        collection.count_documents(
            {}
        )
    )

    print(
        f"Total documents in collection: "
        f"{total_documents}"
    )

    example_document = (
        collection.find_one(
            {},
            {
                "_id": 0,
            },
            sort=[
                (
                    "date",
                    ASCENDING,
                )
            ],
        )
    )

    print(
        "\nExample MongoDB document:"
    )

    print(
        example_document
    )


def main() -> None:
    prediction_results_df = (
        load_prediction_results()
    )

    client, collection = (
        connect_to_collection()
    )

    try:
        create_date_index(
            collection
        )

        upload_prediction_results(
            df=prediction_results_df,
            collection=collection,
        )

        print_summary(
            collection
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()