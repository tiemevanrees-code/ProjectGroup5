
import numpy as np
import pandas as pd
import seaborn as sns

files = [
    ('wind-2017-uur-data.csv', 'wind-2017-uur-data-daily.csv'),
    ('wind-2018-uur-data.csv', 'wind-2018-uur-data-daily.csv'),
    ('wind-2019-uur-data.csv', 'wind-2019-uur-data-daily.csv'),
    ('wind-2020-uur-data.csv', 'wind-2020-uur-data-daily.csv'),
    ('wind-2021-uur-data.csv', 'wind-2021-uur-data-daily.csv'),
]

for input_path, output_path in files:
    df = pd.read_csv(input_path)

    # extract just the date part from the 'validfrom' column
    df['date'] = pd.to_datetime(df['validfrom (UTC)']).dt.date

    # sum numeric columns per day, keep the date
    df_daily = df.groupby('date').sum(numeric_only=True).reset_index()

    # Drop first row (incomplete day)
    df_daily = df_daily.iloc[1:]

    # Keep only the columns we want
    df_daily = df_daily[['date', 'Energy created (kW)']]

    df_daily.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df_daily)} rows)")

