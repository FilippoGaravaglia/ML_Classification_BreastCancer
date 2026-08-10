from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_raw_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wdbc.data"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "eda"


def get_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the 30 predictive numerical features.

    The ID column is excluded because it is an identifier,
    while diagnosis is the target variable.
    """
    return df.drop(columns=["id", "diagnosis"])


def print_descriptive_statistics(features: pd.DataFrame) -> None:
    print("=== DESCRIPTIVE STATISTICS ===")

    statistics = features.describe().T

    print(
        statistics[
            ["mean", "std", "min", "25%", "50%", "75%", "max"]
        ]
    )

    statistics.to_csv(
        RESULTS_DIR / "descriptive_statistics.csv"
    )


def print_feature_ranges(features: pd.DataFrame) -> None:
    print("\n=== FEATURE RANGES ===")

    ranges = pd.DataFrame({
        "min": features.min(),
        "max": features.max(),
    })

    ranges["range"] = ranges["max"] - ranges["min"]

    ranges = ranges.sort_values(
        by="range",
        ascending=False,
    )

    print(ranges)

    ranges.to_csv(
        RESULTS_DIR / "feature_ranges.csv"
    )


def analyze_correlations(features: pd.DataFrame) -> None:
    print("\n=== STRONGLY CORRELATED FEATURE PAIRS ===")

    correlation_matrix = features.corr()

    correlated_pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = correlation_matrix.iloc[i, j]

            if abs(correlation) >= 0.90:
                correlated_pairs.append(
                    (
                        columns[i],
                        columns[j],
                        correlation,
                    )
                )

    correlated_pairs.sort(
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    for feature_1, feature_2, correlation in correlated_pairs:
        print(
            f"{feature_1:30s} "
            f"<-> {feature_2:30s} "
            f"{correlation:.4f}"
        )

    correlation_report = pd.DataFrame(
        correlated_pairs,
        columns=[
            "feature_1",
            "feature_2",
            "correlation",
        ],
    )

    correlation_report.to_csv(
        RESULTS_DIR / "strong_correlations.csv",
        index=False,
    )

    plt.figure(figsize=(14, 12))

    image = plt.imshow(
        correlation_matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(image)

    plt.xticks(
        range(len(columns)),
        columns,
        rotation=90,
        fontsize=6,
    )

    plt.yticks(
        range(len(columns)),
        columns,
        fontsize=6,
    )

    plt.title("Feature Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "correlation_matrix.png",
        dpi=200,
    )

    plt.close()


def analyze_class_means(df: pd.DataFrame) -> None:
    print("\n=== FEATURE MEANS BY DIAGNOSIS ===")

    feature_columns = [
        column
        for column in df.columns
        if column not in ["id", "diagnosis"]
    ]

    class_means = (
        df.groupby("diagnosis")[feature_columns]
        .mean()
        .T
    )

    print(class_means)

    class_means.to_csv(
        RESULTS_DIR / "feature_means_by_diagnosis.csv"
    )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_raw_dataset(DATASET_PATH)

    features = get_feature_dataframe(df)

    print(f"Number of features analyzed: {features.shape[1]}")

    print_descriptive_statistics(features)

    print_feature_ranges(features)

    analyze_correlations(features)

    analyze_class_means(df)

    print(
        "\nFeature analysis completed. "
        f"Results saved in: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()