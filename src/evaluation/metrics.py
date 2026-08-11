from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: str = "M",
    negative_label: str = "B",
) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of observations."
        )

    true_positives = int(
        np.sum(
            (y_true == positive_label)
            & (y_pred == positive_label)
        )
    )

    true_negatives = int(
        np.sum(
            (y_true == negative_label)
            & (y_pred == negative_label)
        )
    )

    false_positives = int(
        np.sum(
            (y_true == negative_label)
            & (y_pred == positive_label)
        )
    )

    false_negatives = int(
        np.sum(
            (y_true == positive_label)
            & (y_pred == negative_label)
        )
    )

    total = len(y_true)

    accuracy = (
        (true_positives + true_negatives) / total
        if total > 0
        else 0.0
    )

    precision_denominator = (
        true_positives + false_positives
    )

    precision = (
        true_positives / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall_denominator = (
        true_positives + false_negatives
    )

    recall = (
        true_positives / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )