from pathlib import Path

import pandas as pd

# Build paths relative to this script, so the code works regardless
# of the PyCharm working-directory setting.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

HISTORICAL_PATH = (
    PROJECT_ROOT / "MVP" / "merged_weather_energy_daily_mvp.csv"
)

FORECAST_2022_PATH = (
    SRC_DIR / "data" / "processed" / "open_meteo_forecast_2022_daily.csv"
)


def inspect_dataset(name: str, path: Path) -> None:
    """
    Print the structure and basic quality checks for one CSV file.
    """
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)
    print(f"Path: {path}")

    if not path.exists():
        print("ERROR: File not found.")
        return

    df = pd.read_csv(path)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nDuplicate dates:")
    if "date" in df.columns:
        print(df["date"].duplicated().sum())
    else:
        print("No lowercase 'date' column found.")


def main() -> None:
    inspect_dataset(
        name="Historical training dataset: 2017–2021",
        path=HISTORICAL_PATH,
    )

    inspect_dataset(
        name="Forecast input dataset: 2022",
        path=FORECAST_2022_PATH,
    )


if __name__ == "__main__":
    main()