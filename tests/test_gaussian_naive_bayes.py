import numpy as np
import pytest

from src.models.gaussian_naive_bayes import (
    GaussianNaiveBayes,
)


def create_simple_dataset():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ]
    )

    y = np.array(
        [
            "B",
            "B",
            "B",
            "M",
            "M",
            "M",
        ]
    )

    return X, y


def test_fit_learns_classes_and_priors():
    X, y = create_simple_dataset()

    model = GaussianNaiveBayes()

    model.fit(X, y)

    assert model.classes_.tolist() == [
        "B",
        "M",
    ]

    np.testing.assert_allclose(
        model.class_priors_,
        [0.5, 0.5],
    )


def test_fit_learns_correct_means():
    X, y = create_simple_dataset()

    model = GaussianNaiveBayes()

    model.fit(X, y)

    expected_means = np.array(
        [
            [2.0, 3.0],
            [9.0, 10.0],
        ]
    )

    np.testing.assert_allclose(
        model.means_,
        expected_means,
    )


def test_fit_learns_positive_variances():
    X, y = create_simple_dataset()

    model = GaussianNaiveBayes()

    model.fit(X, y)

    assert np.all(
        model.variances_ > 0
    )


def test_predict_classifies_clear_examples():
    X, y = create_simple_dataset()

    model = GaussianNaiveBayes()

    model.fit(X, y)

    new_samples = np.array(
        [
            [2.0, 3.0],
            [9.0, 10.0],
        ]
    )

    predictions = model.predict(
        new_samples
    )

    assert predictions.tolist() == [
        "B",
        "M",
    ]


def test_predict_multiple_samples():
    X, y = create_simple_dataset()

    model = GaussianNaiveBayes()

    model.fit(X, y)

    new_samples = np.array(
        [
            [1.5, 2.5],
            [2.5, 3.5],
            [8.5, 9.5],
            [9.5, 10.5],
        ]
    )

    predictions = model.predict(
        new_samples
    )

    assert predictions.tolist() == [
        "B",
        "B",
        "M",
        "M",
    ]


def test_variance_smoothing_handles_constant_feature():
    X = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [8.0, 5.0],
            [9.0, 5.0],
            [10.0, 5.0],
        ]
    )

    y = np.array(
        [
            "B",
            "B",
            "B",
            "M",
            "M",
            "M",
        ]
    )

    model = GaussianNaiveBayes()

    model.fit(X, y)

    assert np.all(
        model.variances_ > 0
    )

    predictions = model.predict(
        np.array(
            [
                [2.0, 5.0],
                [9.0, 5.0],
            ]
        )
    )

    assert predictions.tolist() == [
        "B",
        "M",
    ]


def test_predict_before_fit_raises_error():
    model = GaussianNaiveBayes()

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        model.predict(
            np.array(
                [
                    [1.0, 2.0],
                ]
            )
        )


def test_fit_rejects_different_number_of_samples():
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
        ]
    )

    y = np.array(
        [
            "B",
        ]
    )

    model = GaussianNaiveBayes()

    with pytest.raises(
        ValueError,
        match="same number of observations",
    ):
        model.fit(X, y)