from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent

OUTPUT_PATH = (
    SRC_DIR
    / "data"
    / "model_output"
    / "model_comparison_summary.csv"
)


def main() -> None:
    rows = [
        {
            "model_name": "baseline_linear_regression",
            "evaluation_period": "2021_validation",
            "training_period": "2017-2020",
            "mae_kwh": 8_909_062.29,
            "rmse_kwh": 11_932_533.35,
            "r2": 0.7244,
            "selected_model": False,
        },
        {
            "model_name": "multivariable_linear_regression",
            "evaluation_period": "2021_validation",
            "training_period": "2017-2020",
            "mae_kwh": 6_307_370.11,
            "rmse_kwh": 8_853_440.16,
            "r2": 0.8483,
            "selected_model": True,
        },
        {
            "model_name": "random_forest_regression",
            "evaluation_period": "2021_validation",
            "training_period": "2017-2020",
            "mae_kwh": 7_072_979.13,
            "rmse_kwh": 9_537_212.97,
            "r2": 0.8239,
            "selected_model": False,
        },
        {
            "model_name": "multivariable_linear_regression",
            "evaluation_period": "2022_external",
            "training_period": "2017-2021",
            "mae_kwh": 13_525_311.55,
            "rmse_kwh": 19_763_741.68,
            "r2": 0.5651,
            "selected_model": True,
        },
        {
            "model_name": "random_forest_regression",
            "evaluation_period": "2022_external",
            "training_period": "2017-2021",
            "mae_kwh": 15_683_179.49,
            "rmse_kwh": 19_963_688.09,
            "r2": 0.5563,
            "selected_model": False,
        },
    ]

    comparison_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nModel comparison summary")
    print("-" * 110)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:,.4f}",
        )
    )

    print(
        f"\nSummary saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()