import json
from pathlib import Path

from src.data.loader import load_raw_dataset
from src.data.preprocessing import (
    CORRELATION_THRESHOLD,
    apply_feature_filter,
    combine_features_and_target,
    create_stratified_split,
    find_correlated_features_to_drop,
    prepare_features_and_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wdbc.data"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FULL_FEATURES_DIR = (
    PROCESSED_DIR
    / "full_features"
)

FILTERED_FEATURES_DIR = (
    PROCESSED_DIR
    / "correlation_filtered"
)


def print_class_distribution(
    name: str,
    target,
) -> None:
    print(f"\n=== {name} CLASS DISTRIBUTION ===")

    counts = target.value_counts()
    percentages = (
        target.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    for diagnosis in counts.index:
        print(
            f"{diagnosis}: "
            f"{counts[diagnosis]} "
            f"({percentages[diagnosis]}%)"
        )


def save_dataset(
    directory: Path,
    X_train,
    X_test,
    y_train,
    y_test,
) -> None:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = combine_features_and_target(
        X_train,
        y_train,
    )

    test_df = combine_features_and_target(
        X_test,
        y_test,
    )

    train_df.to_csv(
        directory / "train.csv",
        index=False,
    )

    test_df.to_csv(
        directory / "test.csv",
        index=False,
    )


def main() -> None:
    df = load_raw_dataset(
        RAW_DATASET_PATH
    )

    X, y = prepare_features_and_target(df)

    print("=== PREPROCESSING ===")
    print(f"Original observations: {len(df)}")
    print(f"Predictive features: {X.shape[1]}")
    print("Removed column: id")

    split = create_stratified_split(
        X,
        y,
    )

    print(
        f"\nTraining observations: "
        f"{len(split.X_train)}"
    )

    print(
        f"Test observations: "
        f"{len(split.X_test)}"
    )

    print_class_distribution(
        "TRAIN",
        split.y_train,
    )

    print_class_distribution(
        "TEST",
        split.y_test,
    )

    features_to_drop = (
        find_correlated_features_to_drop(
            split.X_train,
        )
    )

    print(
        "\n=== CORRELATION FILTERING ==="
    )

    print(
        f"Threshold: "
        f"{CORRELATION_THRESHOLD}"
    )

    print(
        f"Features before filtering: "
        f"{split.X_train.shape[1]}"
    )

    print(
        f"Features removed: "
        f"{len(features_to_drop)}"
    )

    for feature in features_to_drop:
        print(f"- {feature}")

    (
        X_train_filtered,
        X_test_filtered,
    ) = apply_feature_filter(
        split.X_train,
        split.X_test,
        features_to_drop,
    )

    print(
        f"Features after filtering: "
        f"{X_train_filtered.shape[1]}"
    )

    save_dataset(
        FULL_FEATURES_DIR,
        split.X_train,
        split.X_test,
        split.y_train,
        split.y_test,
    )

    save_dataset(
        FILTERED_FEATURES_DIR,
        X_train_filtered,
        X_test_filtered,
        split.y_train,
        split.y_test,
    )

    metadata = {
        "test_size": 0.30,
        "random_state": 42,
        "stratified": True,
        "scaling_applied": False,
        "correlation_threshold": (
            CORRELATION_THRESHOLD
        ),
        "original_feature_count": (
            split.X_train.shape[1]
        ),
        "filtered_feature_count": (
            X_train_filtered.shape[1]
        ),
        "removed_features": (
            features_to_drop
        ),
    }

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PROCESSED_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    print(
        "\nScaling applied: NO "
        "(not required for Gaussian Naive Bayes)"
    )

    print(
        "\nProcessed datasets saved successfully."
    )


if __name__ == "__main__":
    main()