import pandas as pd
from pathlib import Path

# Get the directory where this Python file is located
BASE_DIR = Path(__file__).resolve().parent

# File paths
raw_path = BASE_DIR / "KNMI_dataset.csv"
clean_full_path = BASE_DIR / "KNMI_cleaned_daily.csv"
clean_mvp_path = BASE_DIR / "KNMI_cleaned_mvp.csv"

# Load raw KNMI dataset
# The KNMI file uses semicolon as separator
df = pd.read_csv(raw_path, sep=";")

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns to clearer names
df = df.rename(columns={
    "Location": "station_id",
    "Date": "date",
    "DDVEC": "wind_direction_deg",
    "FHVEC": "vector_avg_wind_speed_ms",
    "FG": "avg_wind_speed_ms",
    "FHX": "max_wind_speed_ms",
    "FHXN": "max_wind_hour",
    "FHN": "min_wind_speed_ms",
    "FHNH": "min_wind_hour",
})

# Convert date from YYYYMMDD to real datetime format
df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

# Convert numeric columns
numeric_columns = [
    "station_id",
    "wind_direction_deg",
    "vector_avg_wind_speed_ms",
    "avg_wind_speed_ms",
    "max_wind_speed_ms",
    "max_wind_hour",
    "min_wind_speed_ms",
    "min_wind_hour",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# KNMI wind speed values are stored in 0.1 m/s
# Example: 95 means 9.5 m/s
wind_speed_columns = [
    "vector_avg_wind_speed_ms",
    "avg_wind_speed_ms",
    "max_wind_speed_ms",
    "min_wind_speed_ms",
]

for col in wind_speed_columns:
    df[col] = df[col] / 10

# Save full cleaned dataset
df.to_csv(clean_full_path, index=False)

# Create smaller MVP dataset
# We remove the less important columns for the first MVP:
# max_wind_hour, min_wind_speed_ms, min_wind_hour
mvp_columns = [
    "station_id",
    "date",
    "wind_direction_deg",
    "vector_avg_wind_speed_ms",
    "avg_wind_speed_ms",
]

mvp_df = df[mvp_columns]

# Save MVP cleaned dataset
mvp_df.to_csv(clean_mvp_path, index=False)

# Print results
print("Full cleaned KNMI dataset saved to:", clean_full_path)
print("MVP cleaned KNMI dataset saved to:", clean_mvp_path)

print("\nMVP dataset preview:")
print(mvp_df.head())

print("\nMVP dataset info:")
print(mvp_df.info())

print("\nMissing values in MVP dataset:")
print(mvp_df.isna().sum())