import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.preprocessing import (
    apply_feature_filter,
    find_correlated_features_to_drop,
)
from src.evaluation.metrics import (
    ClassificationMetrics,
    calculate_classification_metrics,
)
from src.models.gaussian_naive_bayes import (
    GaussianNaiveBayes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_features"
    / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "full_features"
    / "test.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "custom"
)

TARGET_COLUMN = "diagnosis"

N_SPLITS = 5
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95


def print_metrics(
    metrics: ClassificationMetrics,
) -> None:
    print(
        f"Accuracy={metrics.accuracy:.4f} | "
        f"Precision={metrics.precision:.4f} | "
        f"Recall={metrics.recall:.4f} | "
        f"F1={metrics.f1:.4f}"
    )


def evaluate_configuration_on_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> ClassificationMetrics:
    model = GaussianNaiveBayes()

    model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
    )

    predictions = model.predict(
        X_validation.to_numpy()
    )

    return calculate_classification_metrics(
        y_validation.to_numpy(),
        predictions,
    )


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    cross_validator = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for fold_number, (
        train_indexes,
        validation_indexes,
    ) in enumerate(
        cross_validator.split(X, y),
        start=1,
    ):
        X_fold_train = X.iloc[
            train_indexes
        ].copy()

        y_fold_train = y.iloc[
            train_indexes
        ].copy()

        X_fold_validation = X.iloc[
            validation_indexes
        ].copy()

        y_fold_validation = y.iloc[
            validation_indexes
        ].copy()

        print(
            f"\n=== FOLD {fold_number} ==="
        )

        print(
            f"Training observations: "
            f"{len(X_fold_train)}"
        )

        print(
            f"Validation observations: "
            f"{len(X_fold_validation)}"
        )

        # FULL CONFIGURATION
        full_metrics = (
            evaluate_configuration_on_fold(
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                y_fold_validation,
            )
        )

        print(
            "\nFULL configuration "
            f"({X_fold_train.shape[1]} features)"
        )

        print_metrics(
            full_metrics
        )

        results.append(
            {
                "fold": fold_number,
                "configuration": "full",
                "feature_count": (
                    X_fold_train.shape[1]
                ),
                "accuracy": (
                    full_metrics.accuracy
                ),
                "precision": (
                    full_metrics.precision
                ),
                "recall": (
                    full_metrics.recall
                ),
                "f1": (
                    full_metrics.f1
                ),
            }
        )

        # FILTERED CONFIGURATION
        features_to_drop = (
            find_correlated_features_to_drop(
                X_fold_train,
                threshold=CORRELATION_THRESHOLD,
            )
        )

        (
            X_fold_train_filtered,
            X_fold_validation_filtered,
        ) = apply_feature_filter(
            X_fold_train,
            X_fold_validation,
            features_to_drop,
        )

        filtered_metrics = (
            evaluate_configuration_on_fold(
                X_fold_train_filtered,
                y_fold_train,
                X_fold_validation_filtered,
                y_fold_validation,
            )
        )

        print(
            "\nFILTERED configuration "
            f"({X_fold_train_filtered.shape[1]} features)"
        )

        print_metrics(
            filtered_metrics
        )

        results.append(
            {
                "fold": fold_number,
                "configuration": "filtered",
                "feature_count": (
                    X_fold_train_filtered.shape[1]
                ),
                "accuracy": (
                    filtered_metrics.accuracy
                ),
                "precision": (
                    filtered_metrics.precision
                ),
                "recall": (
                    filtered_metrics.recall
                ),
                "f1": (
                    filtered_metrics.f1
                ),
            }
        )

    return pd.DataFrame(
        results
    )


def select_best_configuration(
    cv_results: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    summary = (
        cv_results
        .groupby("configuration")
        .agg(
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            feature_count=(
                "feature_count",
                "mean",
            ),
        )
        .reset_index()
    )

    best_row = summary.loc[
        summary["f1"].idxmax()
    ]

    selected_configuration = str(
        best_row["configuration"]
    )

    return (
        selected_configuration,
        summary,
    )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[
        TARGET_COLUMN
    ]

    print(
        "=== CUSTOM GAUSSIAN NAIVE BAYES "
        "EVALUATION ==="
    )

    print(
        f"\nTraining observations: "
        f"{len(X_train)}"
    )

    print(
        f"Final test observations: "
        f"{len(X_test)}"
    )

    print(
        "\nThe final test set is not used "
        "during model selection."
    )

    cv_results = run_cross_validation(
        X_train,
        y_train,
    )

    (
        selected_configuration,
        cv_summary,
    ) = select_best_configuration(
        cv_results
    )

    print(
        "\n\n=== CROSS-VALIDATION SUMMARY ==="
    )

    print(
        cv_summary.round(4).to_string(
            index=False
        )
    )

    print(
        "\nSelected configuration: "
        f"{selected_configuration.upper()}"
    )

    features_to_drop = []

    if selected_configuration == "filtered":
        features_to_drop = (
            find_correlated_features_to_drop(
                X_train,
                threshold=CORRELATION_THRESHOLD,
            )
        )

        (
            X_train_final,
            X_test_final,
        ) = apply_feature_filter(
            X_train,
            X_test,
            features_to_drop,
        )

    else:
        X_train_final = X_train.copy()
        X_test_final = X_test.copy()

    print(
        "\n=== FINAL MODEL TRAINING ==="
    )

    print(
        f"Training observations: "
        f"{len(X_train_final)}"
    )

    print(
        f"Number of features: "
        f"{X_train_final.shape[1]}"
    )

    if features_to_drop:
        print(
            "Removed features:"
        )

        for feature in features_to_drop:
            print(
                f"- {feature}"
            )

    final_model = GaussianNaiveBayes()

    final_model.fit(
        X_train_final.to_numpy(),
        y_train.to_numpy(),
    )

    final_predictions = (
        final_model.predict(
            X_test_final.to_numpy()
        )
    )

    final_metrics = (
        calculate_classification_metrics(
            y_test.to_numpy(),
            final_predictions,
        )
    )

    print(
        "\n=== FINAL TEST RESULTS ==="
    )

    print_metrics(
        final_metrics
    )

    print(
        "\n=== CONFUSION MATRIX COUNTS ==="
    )

    print(
        f"TP = "
        f"{final_metrics.true_positives}"
    )

    print(
        f"TN = "
        f"{final_metrics.true_negatives}"
    )

    print(
        f"FP = "
        f"{final_metrics.false_positives}"
    )

    print(
        f"FN = "
        f"{final_metrics.false_negatives}"
    )

    print(
        "\nCheck: "
        f"TP + TN + FP + FN = "
        f"{final_metrics.true_positives + final_metrics.true_negatives + final_metrics.false_positives + final_metrics.false_negatives}"
    )

    cv_results.to_csv(
        RESULTS_DIR
        / "cross_validation_results.csv",
        index=False,
    )

    cv_summary.to_csv(
        RESULTS_DIR
        / "cross_validation_summary.csv",
        index=False,
    )

    predictions_df = pd.DataFrame(
        {
            "actual": (
                y_test.reset_index(
                    drop=True
                )
            ),
            "predicted": (
                final_predictions
            ),
        }
    )

    predictions_df.to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        index=False,
    )

    with open(
        RESULTS_DIR
        / "final_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(final_metrics),
            file,
            indent=4,
        )

    selection_metadata = {
        "selection_metric": "f1",
        "positive_class": "M",
        "selected_configuration": (
            selected_configuration
        ),
        "correlation_threshold": (
            CORRELATION_THRESHOLD
        ),
        "removed_features": (
            features_to_drop
        ),
        "final_feature_count": (
            X_train_final.shape[1]
        ),
    }

    with open(
        RESULTS_DIR
        / "selection_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection_metadata,
            file,
            indent=4,
        )

    print(
        "\nEvaluation completed successfully."
    )

    print(
        f"Results saved in: "
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()