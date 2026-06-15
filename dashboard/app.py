from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Dutch Wind Energy Dashboard",
    page_icon="🌬️",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_PATH = PROJECT_ROOT / "MVP" / "merged_weather_energy_daily_mvp.csv"
PREDICTION_PATH = PROJECT_ROOT / "src" / "data" / "model_output" / "enriched_prediction_results_2022.csv"
MODEL_SUMMARY_PATH = PROJECT_ROOT / "src" / "data" / "model_output" / "model_comparison_summary.csv"
CAPACITY_PATH = PROJECT_ROOT / "src" / "data" / "raw" / "wind_capacity_by_year.csv"

FORECAST_WEATHER_2022_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "open_meteo_forecast_2022_daily.csv"
ACTUAL_WEATHER_2022_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "knmi_actual_2022_daily.csv"


FEATURE_COLUMNS = [
    "avg_wind_speed_all_stations",
    "vector_avg_wind_speed_all_stations",
    "avg_wind_speed_cubed",
    "wind_direction_sin",
    "wind_direction_cos",
    "installed_wind_capacity_mw",
]

TARGET_COLUMN = "daily_wind_generation_kwh"


COLOR_HISTORICAL = "#2ca02c"
COLOR_ACTUAL = "#1f77b4"
COLOR_PREDICTED = "#ff7f0e"
COLOR_ERROR = "#d62728"
COLOR_ACTUAL_WEATHER = "#9467bd"


@st.cache_data
def load_data():
    historical = pd.read_csv(HISTORICAL_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)
    model_summary = pd.read_csv(MODEL_SUMMARY_PATH)
    capacity = pd.read_csv(CAPACITY_PATH)
    forecast_weather = pd.read_csv(FORECAST_WEATHER_2022_PATH)
    actual_weather = pd.read_csv(ACTUAL_WEATHER_2022_PATH)

    historical["date"] = pd.to_datetime(historical["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])
    forecast_weather["date"] = pd.to_datetime(forecast_weather["date"])
    actual_weather["date"] = pd.to_datetime(actual_weather["date"])

    historical["year"] = historical["date"].dt.year
    predictions["year"] = predictions["date"].dt.year
    forecast_weather["year"] = forecast_weather["date"].dt.year
    actual_weather["year"] = actual_weather["date"].dt.year

    historical = historical.merge(capacity, on="year", how="left")
    forecast_weather = forecast_weather.merge(capacity, on="year", how="left")
    actual_weather = actual_weather.merge(capacity, on="year", how="left")

    return historical, predictions, model_summary, capacity, forecast_weather, actual_weather


def million_kwh(value):
    return f"{value / 1_000_000:,.2f}M kWh"


def create_historical_and_prediction_graph(historical, predictions):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=historical["date"],
            y=historical["daily_wind_generation_kwh"],
            mode="lines",
            name="Actual historical data 2017-2021",
            line=dict(width=2, color=COLOR_HISTORICAL),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["actual_wind_generation_kwh"],
            mode="lines",
            name="Actual 2022 data",
            line=dict(width=2, color=COLOR_ACTUAL),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["predicted_wind_generation_kwh"],
            mode="lines",
            name="Predicted 2022 data",
            line=dict(width=2, dash="dash", color=COLOR_PREDICTED),
        )
    )

    figure.update_layout(
        title="Historical wind generation and 2022 model prediction",
        xaxis_title="Date",
        yaxis_title="Wind generation (kWh)",
        hovermode="x unified",
        height=500,
        legend_title="Data type",
    )

    return figure


def create_actual_vs_predicted_graph(predictions):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["actual_wind_generation_kwh"],
            mode="lines",
            name="Actual 2022",
            line=dict(width=3, color=COLOR_ACTUAL),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["predicted_wind_generation_kwh"],
            mode="lines",
            name="Predicted 2022",
            line=dict(width=3, dash="dash", color=COLOR_PREDICTED),
        )
    )

    figure.update_layout(
        title="Actual 2022 wind generation vs model prediction",
        xaxis_title="Date",
        yaxis_title="Wind generation (kWh)",
        hovermode="x unified",
        height=450,
        legend_title="Generation type",
    )

    return figure


def create_monthly_actual_vs_predicted_barplot(predictions):
    monthly_data = predictions.copy()
    monthly_data["month"] = monthly_data["date"].dt.month
    monthly_data["month_name"] = monthly_data["date"].dt.strftime("%b")

    monthly_summary = (
        monthly_data
        .groupby(["month", "month_name"])
        .agg(
            actual_generation_kwh=("actual_wind_generation_kwh", "sum"),
            predicted_generation_kwh=("predicted_wind_generation_kwh", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=monthly_summary["month_name"],
            y=monthly_summary["actual_generation_kwh"],
            name="Actual generation",
            marker_color=COLOR_ACTUAL,
        )
    )

    figure.add_trace(
        go.Bar(
            x=monthly_summary["month_name"],
            y=monthly_summary["predicted_generation_kwh"],
            name="Predicted generation",
            marker_color=COLOR_PREDICTED,
        )
    )

    figure.update_layout(
        title="Monthly actual vs predicted wind generation in 2022",
        xaxis_title="Month",
        yaxis_title="Wind generation (kWh)",
        barmode="group",
        hovermode="x unified",
        height=500,
        legend_title="Generation type",
    )

    return figure


def create_error_over_time_graph(predictions):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=predictions["date"],
            y=predictions["error_kwh"],
            mode="lines",
            name="Prediction error",
            line=dict(width=2, color=COLOR_ERROR),
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="No error",
    )

    figure.update_layout(
        title="Prediction error over time",
        xaxis_title="Date",
        yaxis_title="Error (kWh)",
        hovermode="x unified",
        height=400,
    )

    return figure


def create_actual_vs_predicted_scatter(predictions):
    figure = px.scatter(
        predictions,
        x="actual_wind_generation_kwh",
        y="predicted_wind_generation_kwh",
        hover_data=["date", "avg_wind_speed_all_stations", "percentage_error"],
        title="Actual vs predicted wind generation",
        labels={
            "actual_wind_generation_kwh": "Actual generation (kWh)",
            "predicted_wind_generation_kwh": "Predicted generation (kWh)",
        },
        color_discrete_sequence=[COLOR_PREDICTED],
    )

    min_value = min(
        predictions["actual_wind_generation_kwh"].min(),
        predictions["predicted_wind_generation_kwh"].min(),
    )

    max_value = max(
        predictions["actual_wind_generation_kwh"].max(),
        predictions["predicted_wind_generation_kwh"].max(),
    )

    figure.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect prediction",
            line=dict(dash="dash", color=COLOR_ACTUAL),
        )
    )

    figure.update_layout(height=450)

    return figure


def create_wind_speed_vs_generation_scatter(historical, predictions):
    historical_scatter = historical[
        [
            "date",
            "avg_wind_speed_all_stations",
            "daily_wind_generation_kwh",
        ]
    ].copy()

    historical_scatter = historical_scatter.rename(
        columns={
            "daily_wind_generation_kwh": "generation_kwh",
        }
    )

    historical_scatter["data_type"] = "Actual 2017-2021"

    prediction_scatter = predictions[
        [
            "date",
            "avg_wind_speed_all_stations",
            "actual_wind_generation_kwh",
        ]
    ].copy()

    prediction_scatter = prediction_scatter.rename(
        columns={
            "actual_wind_generation_kwh": "generation_kwh",
        }
    )

    prediction_scatter["data_type"] = "Actual 2022"

    scatter_data = pd.concat(
        [historical_scatter, prediction_scatter],
        ignore_index=True,
    )

    figure = px.scatter(
        scatter_data,
        x="avg_wind_speed_all_stations",
        y="generation_kwh",
        color="data_type",
        hover_data=["date"],
        title="Average wind speed vs actual wind generation",
        labels={
            "avg_wind_speed_all_stations": "Average wind speed (m/s)",
            "generation_kwh": "Wind generation (kWh)",
            "data_type": "Data type",
        },
        color_discrete_map={
            "Actual 2017-2021": COLOR_HISTORICAL,
            "Actual 2022": COLOR_ACTUAL,
        },
    )

    figure.update_traces(
        marker=dict(
            size=6,
            opacity=0.65,
        )
    )

    figure.update_layout(
        height=500,
        legend_title="Data type",
    )

    return figure


def create_3d_capacity_graph(historical, predictions):
    historical_3d = historical[
        [
            "date",
            "year",
            "avg_wind_speed_all_stations",
            "installed_wind_capacity_mw",
            "daily_wind_generation_kwh",
        ]
    ].copy()

    historical_3d = historical_3d.rename(
        columns={
            "daily_wind_generation_kwh": "generation_kwh",
        }
    )

    historical_3d["data_type"] = "Actual 2017-2021"
    historical_3d["percentage_error"] = None

    prediction_3d = predictions[
        [
            "date",
            "year",
            "avg_wind_speed_all_stations",
            "installed_wind_capacity_mw",
            "predicted_wind_generation_kwh",
            "percentage_error",
        ]
    ].copy()

    prediction_3d = prediction_3d.rename(
        columns={
            "predicted_wind_generation_kwh": "generation_kwh",
        }
    )

    prediction_3d["data_type"] = "Predicted 2022"

    graph_data = pd.concat([historical_3d, prediction_3d], ignore_index=True)

    figure = px.scatter_3d(
        graph_data,
        x="avg_wind_speed_all_stations",
        y="installed_wind_capacity_mw",
        z="generation_kwh",
        color="data_type",
        color_discrete_map={
            "Actual 2017-2021": COLOR_HISTORICAL,
            "Predicted 2022": COLOR_PREDICTED,
        },
        hover_data=["date", "year", "percentage_error"],
        title="3D client view: wind speed, installed capacity and wind generation",
        labels={
            "avg_wind_speed_all_stations": "Average wind speed (m/s)",
            "installed_wind_capacity_mw": "Installed capacity (MW)",
            "generation_kwh": "Wind generation (kWh)",
            "data_type": "Data type",
        },
    )

    figure.update_traces(marker=dict(size=3))

    figure.update_layout(
        height=650,
        scene=dict(
            xaxis_title="Average wind speed (m/s)",
            yaxis_title="Installed capacity (MW)",
            zaxis_title="Wind generation (kWh)",
        ),
    )

    return figure


def prepare_weather_comparison(forecast_weather, actual_weather):
    comparison = actual_weather[
        [
            "date",
            "avg_wind_speed_all_stations",
            "vector_avg_wind_speed_all_stations",
            "avg_wind_speed_cubed",
        ]
    ].copy()

    comparison = comparison.rename(
        columns={
            "avg_wind_speed_all_stations": "actual_avg_wind_speed",
            "vector_avg_wind_speed_all_stations": "actual_vector_wind_speed",
            "avg_wind_speed_cubed": "actual_wind_speed_cubed",
        }
    )

    forecast_part = forecast_weather[
        [
            "date",
            "avg_wind_speed_all_stations",
            "vector_avg_wind_speed_all_stations",
            "avg_wind_speed_cubed",
        ]
    ].copy()

    forecast_part = forecast_part.rename(
        columns={
            "avg_wind_speed_all_stations": "forecast_avg_wind_speed",
            "vector_avg_wind_speed_all_stations": "forecast_vector_wind_speed",
            "avg_wind_speed_cubed": "forecast_wind_speed_cubed",
        }
    )

    comparison = comparison.merge(forecast_part, on="date", how="inner")

    comparison["wind_speed_difference"] = (
        comparison["forecast_avg_wind_speed"]
        - comparison["actual_avg_wind_speed"]
    )

    comparison["wind_speed_percentage_difference"] = (
        comparison["wind_speed_difference"]
        / comparison["actual_avg_wind_speed"]
        * 100
    )

    comparison["wind_speed_cubed_difference"] = (
        comparison["forecast_wind_speed_cubed"]
        - comparison["actual_wind_speed_cubed"]
    )

    return comparison


def create_weather_comparison_graph(weather_comparison):
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=weather_comparison["date"],
            y=weather_comparison["actual_avg_wind_speed"],
            mode="lines",
            name="Actual KNMI wind speed",
            line=dict(width=2, color=COLOR_ACTUAL_WEATHER),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=weather_comparison["date"],
            y=weather_comparison["forecast_avg_wind_speed"],
            mode="lines",
            name="Open-Meteo forecast wind speed",
            line=dict(width=2, dash="dash", color=COLOR_PREDICTED),
        )
    )

    figure.update_layout(
        title="2022 actual KNMI wind speed vs Open-Meteo forecast wind speed",
        xaxis_title="Date",
        yaxis_title="Average wind speed (m/s)",
        hovermode="x unified",
        height=450,
        legend_title="Weather source",
    )

    return figure


def create_weather_difference_graph(weather_comparison):
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=weather_comparison["date"],
            y=weather_comparison["wind_speed_difference"],
            name="Forecast - actual",
            marker_color=COLOR_ERROR,
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="No difference",
    )

    figure.update_layout(
        title="Forecast wind speed error compared with actual KNMI wind speed",
        xaxis_title="Date",
        yaxis_title="Wind speed difference (m/s)",
        height=400,
    )

    return figure


def train_model_and_predict_with_actual_weather(historical, actual_weather, predictions):
    training_data = historical.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    actual_weather = actual_weather.dropna(subset=FEATURE_COLUMNS).copy()

    model = LinearRegression()

    x_train = training_data[FEATURE_COLUMNS]
    y_train = training_data[TARGET_COLUMN]

    model.fit(x_train, y_train)

    actual_weather_predictions = actual_weather[["date"] + FEATURE_COLUMNS].copy()

    actual_weather_predictions["predicted_with_actual_weather_kwh"] = model.predict(
        actual_weather_predictions[FEATURE_COLUMNS]
    )

    actual_values = predictions[
        ["date", "actual_wind_generation_kwh"]
    ].copy()

    result = actual_weather_predictions.merge(actual_values, on="date", how="inner")

    result["error_with_actual_weather_kwh"] = (
        result["predicted_with_actual_weather_kwh"]
        - result["actual_wind_generation_kwh"]
    )

    return result


def create_forecast_vs_actual_weather_prediction_graph(predictions, actual_weather_prediction):
    comparison = predictions[
        [
            "date",
            "actual_wind_generation_kwh",
            "predicted_wind_generation_kwh",
        ]
    ].copy()

    comparison = comparison.merge(
        actual_weather_prediction[
            [
                "date",
                "predicted_with_actual_weather_kwh",
            ]
        ],
        on="date",
        how="inner",
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=comparison["date"],
            y=comparison["actual_wind_generation_kwh"],
            mode="lines",
            name="Actual generation 2022",
            line=dict(width=2, color=COLOR_ACTUAL),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=comparison["date"],
            y=comparison["predicted_wind_generation_kwh"],
            mode="lines",
            name="Prediction with forecast weather",
            line=dict(width=2, dash="dash", color=COLOR_PREDICTED),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=comparison["date"],
            y=comparison["predicted_with_actual_weather_kwh"],
            mode="lines",
            name="Prediction with actual KNMI weather",
            line=dict(width=2, dash="dot", color=COLOR_ACTUAL_WEATHER),
        )
    )

    figure.update_layout(
        title="Effect of weather input on 2022 prediction",
        xaxis_title="Date",
        yaxis_title="Wind generation (kWh)",
        hovermode="x unified",
        height=500,
        legend_title="Prediction type",
    )

    return figure


historical, predictions, model_summary, capacity, forecast_weather, actual_weather = load_data()

weather_comparison = prepare_weather_comparison(forecast_weather, actual_weather)

actual_weather_prediction = train_model_and_predict_with_actual_weather(
    historical,
    actual_weather,
    predictions,
)


st.title("🌬️ Dutch Wind Energy Prediction Dashboard")

st.write(
    """
This dashboard shows how Dutch wind generation developed from 2017 to 2021 and compares it
with the model prediction for 2022. The model used here is the multivariable linear regression model
with weather features and installed wind capacity.
"""
)


st.sidebar.header("Dashboard filter")

show_start = st.sidebar.date_input(
    "Start date",
    value=pd.to_datetime("2017-01-01"),
    min_value=pd.to_datetime("2017-01-01"),
    max_value=pd.to_datetime("2022-12-31"),
)

show_end = st.sidebar.date_input(
    "End date",
    value=pd.to_datetime("2022-12-31"),
    min_value=pd.to_datetime("2017-01-01"),
    max_value=pd.to_datetime("2022-12-31"),
)

if show_start > show_end:
    st.error("Start date cannot be later than end date.")
    st.stop()


historical_filtered = historical[
    (historical["date"] >= pd.to_datetime(show_start))
    & (historical["date"] <= pd.to_datetime(show_end))
]

predictions_filtered = predictions[
    (predictions["date"] >= pd.to_datetime(show_start))
    & (predictions["date"] <= pd.to_datetime(show_end))
]

weather_comparison_filtered = weather_comparison[
    (weather_comparison["date"] >= pd.to_datetime(show_start))
    & (weather_comparison["date"] <= pd.to_datetime(show_end))
]

actual_weather_prediction_filtered = actual_weather_prediction[
    (actual_weather_prediction["date"] >= pd.to_datetime(show_start))
    & (actual_weather_prediction["date"] <= pd.to_datetime(show_end))
]


mae = mean_absolute_error(
    predictions["actual_wind_generation_kwh"],
    predictions["predicted_wind_generation_kwh"],
)

rmse = mean_squared_error(
    predictions["actual_wind_generation_kwh"],
    predictions["predicted_wind_generation_kwh"],
) ** 0.5

r2_forecast_weather = r2_score(
    predictions["actual_wind_generation_kwh"],
    predictions["predicted_wind_generation_kwh"],
)

r2_actual_weather = r2_score(
    actual_weather_prediction["actual_wind_generation_kwh"],
    actual_weather_prediction["predicted_with_actual_weather_kwh"],
)

mae_actual_weather = mean_absolute_error(
    actual_weather_prediction["actual_wind_generation_kwh"],
    actual_weather_prediction["predicted_with_actual_weather_kwh"],
)

average_bias = predictions["error_kwh"].mean()

forecast_wind_speed_mean = weather_comparison["forecast_avg_wind_speed"].mean()
actual_wind_speed_mean = weather_comparison["actual_avg_wind_speed"].mean()

wind_speed_bias_percentage = (
    (forecast_wind_speed_mean - actual_wind_speed_mean)
    / actual_wind_speed_mean
    * 100
)

forecast_lower_days = (
    weather_comparison["forecast_avg_wind_speed"]
    < weather_comparison["actual_avg_wind_speed"]
).sum()


st.subheader("Model performance summary for 2022")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "Multivariable LR")
col2.metric("2022 R² score", f"{r2_forecast_weather:.3f}")
col3.metric("MAE", million_kwh(mae))
col4.metric("Average bias", million_kwh(average_bias))

st.info(
    """
The 2022 R² shown above is based on the realistic forecast setup: the model predicts 2022 wind
generation using Open-Meteo forecast weather. This means the prediction quality depends not only on
the model, but also on the quality of the weather forecast input.
"""
)


with st.expander("Show model comparison table"):
    st.dataframe(model_summary, use_container_width=True)


st.subheader("Historical data and model prediction")

st.plotly_chart(
    create_historical_and_prediction_graph(historical_filtered, predictions_filtered),
    use_container_width=True,
)

st.write(
    """
This graph is the main client graph. It shows the real historical wind generation from 2017 to 2021.
After that, it shows the actual 2022 generation and the model prediction for 2022.
"""
)


st.subheader("Actual 2022 vs predicted 2022")

st.plotly_chart(
    create_actual_vs_predicted_graph(predictions_filtered),
    use_container_width=True,
)


st.subheader("Monthly actual vs predicted wind generation")

st.plotly_chart(
    create_monthly_actual_vs_predicted_barplot(predictions_filtered),
    use_container_width=True,
)

st.write(
    """
This monthly bar chart summarizes the daily 2022 results into monthly totals. It makes the
underprediction easier to see than in the daily line graph. If the predicted bars are consistently lower
than the actual bars, this supports the conclusion that the model underestimates 2022 wind generation.
"""
)


st.subheader("Weather input quality check")

weather_col1, weather_col2, weather_col3, weather_col4 = st.columns(4)

weather_col1.metric(
    "Actual KNMI wind speed",
    f"{actual_wind_speed_mean:.2f} m/s",
)

weather_col2.metric(
    "Forecast wind speed",
    f"{forecast_wind_speed_mean:.2f} m/s",
)

weather_col3.metric(
    "Forecast bias",
    f"{wind_speed_bias_percentage:.1f}%",
)

weather_col4.metric(
    "Forecast lower days",
    f"{forecast_lower_days} / {len(weather_comparison)}",
)

st.plotly_chart(
    create_weather_comparison_graph(weather_comparison_filtered),
    use_container_width=True,
)

st.plotly_chart(
    create_weather_difference_graph(weather_comparison_filtered),
    use_container_width=True,
)

st.write(
    """
The Open-Meteo forecast wind speed is compared with the actual measured KNMI wind speed for 2022.
If the forecast wind speed is consistently lower than the actual wind speed, the model will also predict
lower wind generation. This is especially important because wind speed cubed is used as a feature,
which makes underestimation in wind speed even stronger in the model input.
"""
)


st.subheader("Effect of weather input on prediction quality")

input_col1, input_col2, input_col3 = st.columns(3)

input_col1.metric(
    "R² with forecast weather",
    f"{r2_forecast_weather:.3f}",
)

input_col2.metric(
    "R² with actual KNMI weather",
    f"{r2_actual_weather:.3f}",
)

input_col3.metric(
    "MAE with actual weather",
    million_kwh(mae_actual_weather),
)

st.plotly_chart(
    create_forecast_vs_actual_weather_prediction_graph(
        predictions_filtered,
        actual_weather_prediction_filtered,
    ),
    use_container_width=True,
)

st.success(
    """
This test keeps the model the same, but changes the 2022 weather input from forecast weather to
actual KNMI weather. If the R² improves, this shows that part of the low 2022 score is caused by
forecast-weather bias, not only by the model itself.
"""
)


st.subheader("Wind speed relationship with generation")

st.plotly_chart(
    create_wind_speed_vs_generation_scatter(
        historical_filtered,
        predictions_filtered,
    ),
    use_container_width=True,
)

st.write(
    """
This scatterplot shows the relationship between average wind speed and actual wind generation.
The upward pattern explains why wind speed is an important model feature. The spread in the points
also shows that wind speed alone cannot explain everything.
"""
)


st.subheader("Error analysis")

error_col1, error_col2 = st.columns(2)

with error_col1:
    st.plotly_chart(
        create_error_over_time_graph(predictions_filtered),
        use_container_width=True,
    )

with error_col2:
    st.plotly_chart(
        create_actual_vs_predicted_scatter(predictions_filtered),
        use_container_width=True,
    )

st.write(
    """
The error graphs show where the model performs well and where it struggles. A perfect model would
have points close to the diagonal line in the scatter plot. Large deviations show days where the model
overestimated or underestimated wind generation.
"""
)


st.subheader("Interactive 3D client graph")

st.write(
    """
This 3D graph shows the relationship between average wind speed, installed wind capacity and wind
generation. Because installed capacity changes per year, the graph forms several capacity layers.
That helps explain why production can increase over time even when wind speed is similar.
"""
)

st.plotly_chart(
    create_3d_capacity_graph(historical_filtered, predictions_filtered),
    use_container_width=True,
)


with st.expander("Show dashboard data"):
    st.write("Historical data")
    st.dataframe(historical_filtered, use_container_width=True)

    st.write("Prediction data")
    st.dataframe(predictions_filtered, use_container_width=True)

    st.write("Weather comparison data")
    st.dataframe(weather_comparison_filtered, use_container_width=True)

    st.write("Prediction using actual KNMI weather")
    st.dataframe(actual_weather_prediction_filtered, use_container_width=True)