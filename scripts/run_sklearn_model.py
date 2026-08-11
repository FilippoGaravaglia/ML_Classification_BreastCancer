import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import GaussianNB

from src.evaluation.metrics import (
    calculate_classification_metrics,
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
    / "sklearn"
)

TARGET_COLUMN = "diagnosis"


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
        "=== SCIKIT-LEARN GAUSSIAN NAIVE BAYES ==="
    )

    print(
        f"\nTraining observations: "
        f"{len(X_train)}"
    )

    print(
        f"Test observations: "
        f"{len(X_test)}"
    )

    print(
        f"Number of features: "
        f"{X_train.shape[1]}"
    )

    model = GaussianNB(
        var_smoothing=1e-9
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    metrics = (
        calculate_classification_metrics(
            y_test.to_numpy(),
            predictions,
        )
    )

    print(
        "\n=== LEARNED MODEL ==="
    )

    print(
        f"Classes learned: "
        f"{model.classes_.tolist()}"
    )

    print(
        f"Class priors: "
        f"{model.class_prior_.tolist()}"
    )

    print(
        f"Means shape: "
        f"{model.theta_.shape}"
    )

    print(
        f"Variances shape: "
        f"{model.var_.shape}"
    )

    print(
        "\n=== FINAL TEST RESULTS ==="
    )

    print(
        f"Accuracy : "
        f"{metrics.accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics.precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics.recall:.4f}"
    )

    print(
        f"F1-score : "
        f"{metrics.f1:.4f}"
    )

    print(
        "\n=== CONFUSION MATRIX COUNTS ==="
    )

    print(
        f"TP = "
        f"{metrics.true_positives}"
    )

    print(
        f"TN = "
        f"{metrics.true_negatives}"
    )

    print(
        f"FP = "
        f"{metrics.false_positives}"
    )

    print(
        f"FN = "
        f"{metrics.false_negatives}"
    )

    print(
        "\nCheck: "
        f"TP + TN + FP + FN = "
        f"{metrics.true_positives + metrics.true_negatives + metrics.false_positives + metrics.false_negatives}"
    )

    predictions_df = pd.DataFrame(
        {
            "actual": (
                y_test.reset_index(
                    drop=True
                )
            ),
            "predicted": predictions,
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
            asdict(metrics),
            file,
            indent=4,
        )

    model_metadata = {
        "library": "scikit-learn",
        "model": "GaussianNB",
        "configuration": "full",
        "feature_count": (
            X_train.shape[1]
        ),
        "var_smoothing": 1e-9,
        "training_observations": (
            len(X_train)
        ),
        "test_observations": (
            len(X_test)
        ),
    }

    with open(
        RESULTS_DIR
        / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_metadata,
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