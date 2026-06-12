from datetime import date

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Dutch Wind Energy Dashboard",
    page_icon="🌬️",
    layout="wide",
)

st.title("Dutch Wind Energy Prediction Dashboard")

st.write(
    "This dashboard shows predicted and actual Dutch wind-energy "
    "generation for 2022 using the selected multivariable linear-regression model."
)


@st.cache_data(ttl=60)
def get_summary(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Retrieve aggregated model-performance statistics from the API.
    """
    params = {}

    if start_date is not None:
        params["start_date"] = start_date

    if end_date is not None:
        params["end_date"] = end_date

    response = requests.get(
        f"{API_BASE_URL}/summary",
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=60)
def get_predictions(
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Retrieve daily prediction records from the API.
    """
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "limit": 365,
    }

    response = requests.get(
        f"{API_BASE_URL}/predictions",
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=60)
def get_prediction_by_date(
    selected_date: str,
) -> dict:
    """
    Retrieve one daily prediction record from the API.
    """
    response = requests.get(
        f"{API_BASE_URL}/predictions/{selected_date}",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def convert_records_to_dataframe(
    records: list[dict],
) -> pd.DataFrame:
    """
    Convert nested API records into a flat dataframe for visualization.
    """
    rows = []

    for record in records:
        rows.append(
            {
                "date": record["date"],
                "avg_wind_speed_ms": record[
                    "weather"
                ][
                    "avg_wind_speed_all_stations"
                ],
                "avg_wind_speed_cubed": record[
                    "weather"
                ][
                    "avg_wind_speed_cubed"
                ],
                "wind_direction_sin": record[
                    "weather"
                ][
                    "wind_direction_sin"
                ],
                "wind_direction_cos": record[
                    "weather"
                ][
                    "wind_direction_cos"
                ],
                "installed_capacity_mw": record[
                    "installed_wind_capacity_mw"
                ],
                "predicted_generation_kwh": record[
                    "prediction"
                ][
                    "predicted_wind_generation_kwh"
                ],
                "actual_generation_kwh": record[
                    "prediction"
                ][
                    "actual_wind_generation_kwh"
                ],
                "error_kwh": record[
                    "prediction"
                ][
                    "error_kwh"
                ],
                "absolute_error_kwh": record[
                    "prediction"
                ][
                    "absolute_error_kwh"
                ],
                "percentage_error": record[
                    "prediction"
                ][
                    "percentage_error"
                ],
            }
        )

    df = pd.DataFrame(
        rows
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def format_million_kwh(
    value: float,
) -> str:
    """
    Format a kWh value as millions of kWh.
    """
    return (
        f"{value / 1_000_000:,.2f} million kWh"
    )


st.sidebar.header(
    "Filters"
)

start_date = st.sidebar.date_input(
    "Start date",
    value=date(
        2022,
        1,
        1,
    ),
    min_value=date(
        2022,
        1,
        1,
    ),
    max_value=date(
        2022,
        12,
        31,
    ),
)

end_date = st.sidebar.date_input(
    "End date",
    value=date(
        2022,
        12,
        31,
    ),
    min_value=date(
        2022,
        1,
        1,
    ),
    max_value=date(
        2022,
        12,
        31,
    ),
)

if start_date > end_date:
    st.error(
        "The start date cannot be later than the end date."
    )

    st.stop()

start_date_string = start_date.isoformat()
end_date_string = end_date.isoformat()

try:
    summary = get_summary(
        start_date=start_date_string,
        end_date=end_date_string,
    )

    prediction_records = get_predictions(
        start_date=start_date_string,
        end_date=end_date_string,
    )

except requests.RequestException as error:
    st.error(
        "The dashboard could not connect to the FastAPI backend. "
        "Start the API with: uvicorn api.main:app --reload"
    )

    st.exception(
        error
    )

    st.stop()

df = convert_records_to_dataframe(
    prediction_records
)

st.header(
    "Model overview"
)

column_1, column_2, column_3, column_4 = (
    st.columns(
        4
    )
)

column_1.metric(
    "Selected model",
    "Linear regression",
)

column_2.metric(
    "R² score",
    f"{summary['r2']:.3f}",
)

column_3.metric(
    "MAE",
    format_million_kwh(
        summary[
            "mae_kwh"
        ]
    ),
)

column_4.metric(
    "RMSE",
    format_million_kwh(
        summary[
            "rmse_kwh"
        ]
    ),
)

column_5, column_6, column_7, column_8 = (
    st.columns(
        4
    )
)

column_5.metric(
    "Average actual generation",
    format_million_kwh(
        summary[
            "average_actual_generation_kwh"
        ]
    ),
)

column_6.metric(
    "Average predicted generation",
    format_million_kwh(
        summary[
            "average_predicted_generation_kwh"
        ]
    ),
)

column_7.metric(
    "Average prediction bias",
    format_million_kwh(
        summary[
            "average_prediction_bias_kwh"
        ]
    ),
)

column_8.metric(
    "Correlation",
    f"{summary['correlation']:.3f}",
)

st.caption(
    "A negative prediction bias means that the model tends "
    "to underestimate actual wind-energy generation."
)

st.header(
    "Actual versus predicted wind-energy generation"
)

line_chart_df = (
    df[
        [
            "date",
            "actual_generation_kwh",
            "predicted_generation_kwh",
        ]
    ]
    .set_index(
        "date"
    )
    .rename(
        columns={
            "actual_generation_kwh":
                "Actual generation",
            "predicted_generation_kwh":
                "Predicted generation",
        }
    )
)

st.line_chart(
    line_chart_df
)

st.header(
    "Prediction error over time"
)

error_chart_df = (
    df[
        [
            "date",
            "error_kwh",
        ]
    ]
    .set_index(
        "date"
    )
    .rename(
        columns={
            "error_kwh":
                "Prediction error",
        }
    )
)

st.line_chart(
    error_chart_df
)

st.caption(
    "Negative values mean that the model underestimated "
    "the actual wind-energy generation."
)

st.header(
    "Relationship between wind speed and generation"
)

scatter_df = (
    df[
        [
            "avg_wind_speed_ms",
            "actual_generation_kwh",
        ]
    ]
    .rename(
        columns={
            "avg_wind_speed_ms":
                "Average wind speed (m/s)",
            "actual_generation_kwh":
                "Actual generation (kWh)",
        }
    )
)

st.scatter_chart(
    scatter_df,
    x="Average wind speed (m/s)",
    y="Actual generation (kWh)",
)

st.header(
    "Investigate one day"
)

selected_date = st.date_input(
    "Select a date",
    value=date(
        2022,
        1,
        1,
    ),
    min_value=date(
        2022,
        1,
        1,
    ),
    max_value=date(
        2022,
        12,
        31,
    ),
)

try:
    selected_day = get_prediction_by_date(
        selected_date.isoformat()
    )

except requests.RequestException as error:
    st.error(
        "The selected daily result could not be retrieved."
    )

    st.exception(
        error
    )

    st.stop()

daily_column_1, daily_column_2, daily_column_3 = (
    st.columns(
        3
    )
)

daily_column_1.metric(
    "Average wind speed",
    (
        f"{selected_day['weather']['avg_wind_speed_all_stations']:.2f} m/s"
    ),
)

daily_column_2.metric(
    "Installed wind capacity",
    (
        f"{selected_day['installed_wind_capacity_mw']:,} MW"
    ),
)

daily_column_3.metric(
    "Percentage error",
    (
        f"{selected_day['prediction']['percentage_error']:.2f}%"
    ),
)

daily_column_4, daily_column_5, daily_column_6 = (
    st.columns(
        3
    )
)

daily_column_4.metric(
    "Predicted generation",
    format_million_kwh(
        selected_day[
            "prediction"
        ][
            "predicted_wind_generation_kwh"
        ]
    ),
)

daily_column_5.metric(
    "Actual generation",
    format_million_kwh(
        selected_day[
            "prediction"
        ][
            "actual_wind_generation_kwh"
        ]
    ),
)

daily_column_6.metric(
    "Prediction error",
    format_million_kwh(
        selected_day[
            "prediction"
        ][
            "error_kwh"
        ]
    ),
)

st.header(
    "Largest error in selected period"
)

st.write(
    (
        f"The largest absolute prediction error occurred on "
        f"**{summary['largest_absolute_error']['date']}** "
        f"and was "
        f"**{format_million_kwh(summary['largest_absolute_error']['absolute_error_kwh'])}**."
    )
)

st.header(
    "Daily results table"
)

table_df = (
    df[
        [
            "date",
            "avg_wind_speed_ms",
            "installed_capacity_mw",
            "predicted_generation_kwh",
            "actual_generation_kwh",
            "error_kwh",
            "percentage_error",
        ]
    ]
    .copy()
)

st.dataframe(
    table_df,
    use_container_width=True,
)

st.header(
    "Interpretation and limitations"
)

st.write(
    """
The model uses average wind speed, wind speed cubed, transformed wind-direction
features, and installed wind capacity to estimate daily Dutch wind-energy generation.

The dashboard should be interpreted as an analytical prototype rather than a
production forecasting system. The model still underestimates generation on many
days. One likely reason is that annual installed-capacity values are assigned to every
day within the same year, even though new wind farms may have become operational
gradually. The historical KNMI observations and archived Open-Meteo forecast inputs
may also differ in measurement method and spatial coverage.
"""
)