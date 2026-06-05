import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

mongodb_uri = os.getenv("MONGODB_URI")

if not mongodb_uri:
    raise ValueError(
        "MONGODB_URI was not found. Check your .env file."
    )

client = MongoClient(
    mongodb_uri,
    serverSelectionTimeoutMS=5000,
)

try:
    client.admin.command("ping")

    print("MongoDB connection successful.")

    database = client["wind_energy_project"]
    collection = database["connection_test"]

    test_document = {
        "message": "MongoDB connection works"
    }

    inserted_result = collection.insert_one(test_document)

    print("Test document inserted successfully.")
    print(f"Inserted document ID: {inserted_result.inserted_id}")

    collection.delete_one(
        {"_id": inserted_result.inserted_id}
    )

    print("Test document removed successfully.")

finally:
    client.close()