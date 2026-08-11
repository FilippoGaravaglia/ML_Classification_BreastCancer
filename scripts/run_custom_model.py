from pathlib import Path

import pandas as pd

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

TARGET_COLUMN = "diagnosis"


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

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

    model = GaussianNaiveBayes()

    model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
    )

    predictions = model.predict(
        X_test.to_numpy()
    )

    print("=== CUSTOM GAUSSIAN NAIVE BAYES ===")

    print(
        f"Training observations: "
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

    print(
        f"Classes learned: "
        f"{model.classes_.tolist()}"
    )

    print("\n=== CLASS PRIORS ===")

    for class_label, prior in zip(
        model.classes_,
        model.class_priors_,
    ):
        print(
            f"P({class_label}) = "
            f"{prior:.4f}"
        )

    print(
        "\n=== LEARNED PARAMETER SHAPES ==="
    )

    print(
        f"Means shape: "
        f"{model.means_.shape}"
    )

    print(
        f"Variances shape: "
        f"{model.variances_.shape}"
    )

    print("\n=== FIRST 10 PREDICTIONS ===")

    for index in range(
        min(10, len(predictions))
    ):
        print(
            f"Sample {index + 1:2d}: "
            f"predicted={predictions[index]} "
            f"actual={y_test.iloc[index]}"
        )

    print(
        "\nModel training and prediction "
        "completed successfully."
    )


if __name__ == "__main__":
    main()