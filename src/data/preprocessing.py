from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"

TEST_SIZE = 0.30
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.95


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictive features from the target.

    The ID column is removed because it only identifies the observation
    and has no predictive meaning.
    """
    X = df.drop(
        columns=[ID_COLUMN, TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].copy()

    return X, y


def create_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> DatasetSplit:
    """
    Split the dataset into 70% training data and 30% test data.

    Stratification preserves approximately the same class proportions
    in both subsets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def find_correlated_features_to_drop(
    X_train: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> list[str]:
    """
    Find redundant features using only the training set.

    If a feature has an absolute Pearson correlation greater than or
    equal to the threshold with an earlier feature, it is marked for
    removal.
    """
    correlation_matrix = X_train.corr().abs()

    columns_to_drop: list[str] = []

    columns = list(correlation_matrix.columns)

    for current_index in range(len(columns)):
        current_feature = columns[current_index]

        for previous_index in range(current_index):
            previous_feature = columns[previous_index]

            correlation = correlation_matrix.loc[
                current_feature,
                previous_feature,
            ]

            if correlation >= threshold:
                columns_to_drop.append(current_feature)
                break

    return columns_to_drop


def apply_feature_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features_to_drop: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the feature selection learned from the training set
    to both training and test data.
    """
    X_train_filtered = X_train.drop(
        columns=features_to_drop
    )

    X_test_filtered = X_test.drop(
        columns=features_to_drop
    )

    return X_train_filtered, X_test_filtered


def combine_features_and_target(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Recombine features and target into a single dataframe.

    Diagnosis is deliberately kept as the last column, which also
    simplifies later use in WEKA.
    """
    combined = X.copy()

    combined[TARGET_COLUMN] = y.values

    return combined