from datetime import date

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

MODEL_DISPLAY_NAME = "Multivariable Linear Regression"
MODEL_VERSION = "v1_capacity_adjusted"


st.set_page_config(
    page_title="Dutch Wind Energy Dashboard",
    page_icon="🌬️",
    layout="wide",
)


def get_summary(start_date: str, end_date: str) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/summary",
        params={
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_predictions(start_date: str, end_date: str) -> list[dict]:
    response = requests.get(
        f"{API_BASE_URL}/predictions",
        params={
            "start_date": start_date,
            "end_date": end_date,
            "limit": 365,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []

    for record in records:
        rows.append(
            {
                "date": record["date"],
                "avg_wind_speed_ms": record["weather"]["avg_wind_speed_all_stations"],
                "avg_wind_speed_cubed": record["weather"]["avg_wind_speed_cubed"],
                "installed_capacity_mw": record["installed_wind_capacity_mw"],
                "actual_kwh": record["prediction"]["actual_wind_generation_kwh"],
                "predicted_kwh": record["prediction"]["predicted_wind_generation_kwh"],
                "error_kwh": record["prediction"]["error_kwh"],
                "absolute_error_kwh": record["prediction"]["absolute_error_kwh"],
                "percentage_error": record["prediction"]["percentage_error"],
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    return df


def format_million_kwh(value: float) -> str:
    return f"{value / 1_000_000:,.2f}M kWh"


def create_actual_vs_prediction_graph(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["actual_kwh"],
            mode="lines",
            name="Actual 2022 generation",
            line=dict(width=2),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["predicted_kwh"],
            mode="lines",
            name="Predicted generation",
            line=dict(width=2, dash="dash"),
        )
    )

    figure.update_layout(
        title="Actual 2022 wind generation vs model prediction",
        xaxis_title="Date",
        yaxis_title="Wind energy generation (kWh)",
        hovermode="x unified",
        height=450,
    )

    return figure


def create_3d_graph(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter3d(
            x=df["avg_wind_speed_ms"],
            y=df["month"],
            z=df["actual_kwh"],
            mode="markers",
            marker=dict(
                size=5,
                color=df["percentage_error"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Error (%)"),
            ),
            text=df["date"].dt.strftime("%Y-%m-%d"),
            hovertemplate=(
                "Date: %{text}<br>"
                "Wind speed: %{x:.2f} m/s<br>"
                "Month: %{y}<br>"
                "Actual generation: %{z:,.0f} kWh<br>"
                "Error: %{marker.color:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="3D exploration: wind speed, month and actual generation",
        scene=dict(
            xaxis_title="Average wind speed (m/s)",
            yaxis_title="Month",
            zaxis_title="Actual generation (kWh)",
        ),
        height=600,
    )

    return figure


st.title("🌬️ Dutch Wind Energy Prediction Dashboard")

st.write(
    "This dashboard shows the selected **multivariable linear regression** model. "
    "It compares actual Dutch wind generation in 2022 with the model prediction "
    "and lets users explore the relationship between weather, season, and energy generation."
)

st.info(
    f"Selected model: **{MODEL_DISPLAY_NAME}**  |  Model version: **{MODEL_VERSION}**"
)


st.sidebar.header("Date filter")

start_date = st.sidebar.date_input(
    "Start date",
    value=date(2022, 1, 1),
    min_value=date(2022, 1, 1),
    max_value=date(2022, 12, 31),
)

end_date = st.sidebar.date_input(
    "End date",
    value=date(2022, 12, 31),
    min_value=date(2022, 1, 1),
    max_value=date(2022, 12, 31),
)

if start_date > end_date:
    st.error("Start date cannot be later than end date.")
    st.stop()

start_date_string = start_date.isoformat()
end_date_string = end_date.isoformat()


try:
    summary = get_summary(
        start_date=start_date_string,
        end_date=end_date_string,
    )

    records = get_predictions(
        start_date=start_date_string,
        end_date=end_date_string,
    )

except requests.RequestException as error:
    st.error(
        "Could not connect to the FastAPI backend. "
        "Start it first with: uvicorn api.main:app --reload"
    )
    st.exception(error)
    st.stop()


df = records_to_dataframe(records)


st.subheader("Model performance summary")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Model",
    "Multivariable LR",
)

metric_2.metric(
    "R² score",
    f"{summary['r2']:.3f}",
)

metric_3.metric(
    "Mean absolute error",
    format_million_kwh(summary["mae_kwh"]),
)

metric_4.metric(
    "Average bias",
    format_million_kwh(summary["average_prediction_bias_kwh"]),
)

st.caption(
    "A negative prediction bias means that the model usually predicts less generation "
    "than was actually produced."
)


st.subheader("Actual 2022 generation and model prediction")

st.plotly_chart(
    create_actual_vs_prediction_graph(df),
    use_container_width=True,
)


st.subheader("Interactive 3D data exploration")

st.write(
    "This 3D plot shows how wind speed and month relate to actual wind-energy generation. "
    "The colour of each point shows the model's percentage error for that day."
)

st.plotly_chart(
    create_3d_graph(df),
    use_container_width=True,
)


st.subheader("Short interpretation")

st.write(
    """
The line graph shows that the multivariable linear regression model follows the general
movement of wind-energy generation, but it still underestimates many days.

The 3D graph helps users explore the relationship between wind speed, season, and actual
generation. Days with higher wind speeds generally produce more wind energy, but the
prediction error shows that weather alone does not explain everything. Installed capacity,
regional differences, turbine availability, curtailment, and the difference between KNMI
observations and Open-Meteo forecast inputs can also affect the results.
"""
)


with st.expander("Show daily data table"):
    table_df = df[
        [
            "date",
            "avg_wind_speed_ms",
            "installed_capacity_mw",
            "actual_kwh",
            "predicted_kwh",
            "error_kwh",
            "percentage_error",
        ]
    ].copy()

    table_df = table_df.rename(
        columns={
            "date": "Date",
            "avg_wind_speed_ms": "Average wind speed (m/s)",
            "installed_capacity_mw": "Installed capacity (MW)",
            "actual_kwh": "Actual generation (kWh)",
            "predicted_kwh": "Predicted generation (kWh)",
            "error_kwh": "Prediction error (kWh)",
            "percentage_error": "Percentage error (%)",
        }
    )

    st.dataframe(
        table_df,
        use_container_width=True,
    )