import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented from scratch.

    The model estimates, for each class and feature:
    - the class prior probability;
    - the feature mean;
    - the feature variance.

    Predictions are computed using Gaussian log-likelihoods.
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
    ) -> None:
        self.var_smoothing = var_smoothing

        self.classes_: np.ndarray | None = None
        self.class_priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.epsilon_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "GaussianNaiveBayes":
        """
        Learn class priors, means and variances from the training data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional array."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of observations."
            )

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_priors_ = np.zeros(
            n_classes,
            dtype=float,
        )

        self.means_ = np.zeros(
            (n_classes, n_features),
            dtype=float,
        )

        self.variances_ = np.zeros(
            (n_classes, n_features),
            dtype=float,
        )

        global_variance = np.var(
            X,
            axis=0,
        )

        self.epsilon_ = (
            self.var_smoothing
            * np.max(global_variance)
        )

        for class_index, class_label in enumerate(
            self.classes_
        ):
            X_class = X[y == class_label]

            self.class_priors_[class_index] = (
                len(X_class) / len(X)
            )

            self.means_[class_index] = np.mean(
                X_class,
                axis=0,
            )

            self.variances_[class_index] = (
                np.var(
                    X_class,
                    axis=0,
                )
                + self.epsilon_
            )

        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the most probable class for each observation.
        """
        self._check_is_fitted()

        X = np.asarray(
            X,
            dtype=float,
        )

        joint_log_likelihood = (
            self._calculate_joint_log_likelihood(X)
        )

        class_indexes = np.argmax(
            joint_log_likelihood,
            axis=1,
        )

        return self.classes_[class_indexes]

    def _calculate_joint_log_likelihood(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate one log-probability score for each
        observation and each possible class.
        """
        scores = []

        for class_index in range(
            len(self.classes_)
        ):
            log_prior = np.log(
                self.class_priors_[class_index]
            )

            mean = self.means_[class_index]
            variance = self.variances_[class_index]

            log_likelihood = (
                -0.5
                * np.log(
                    2.0 * np.pi * variance
                )
                - (
                    (X - mean) ** 2
                    / (2.0 * variance)
                )
            )

            total_log_likelihood = np.sum(
                log_likelihood,
                axis=1,
            )

            scores.append(
                log_prior
                + total_log_likelihood
            )

        return np.column_stack(scores)

    def _check_is_fitted(self) -> None:
        """
        Ensure that fit() has been called before prediction.
        """
        if (
            self.classes_ is None
            or self.class_priors_ is None
            or self.means_ is None
            or self.variances_ is None
        ):
            raise RuntimeError(
                "The model must be fitted before prediction."
            )