"""Conditional fractional differencing with zero pre-sample residuals."""

from numbers import Real

import numpy as np


class FractionalFilter:
    """Finite-history (1-L)**d filter and its derivatives with respect to d."""

    lower_bound = -0.5 + 1e-8
    upper_bound = 0.5 - 1e-8

    def __init__(self, d: float, length: int):
        if (not isinstance(d, Real) or isinstance(d, (bool, np.bool_))
                or not np.isfinite(d) or not -0.5 < d < 0.5):
            raise ValueError("fractional d must be a finite real number in (-0.5, 0.5)")
        if not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("fractional filter length must be a positive integer")
        self.d = float(d)
        self.weights = np.zeros(length)
        self.first = np.zeros(length)
        self.second = np.zeros(length)
        self.weights[0] = 1.0
        # Differentiate the recurrence directly; this also works at d=0.
        for lag in range(1, length):
            factor = (lag - 1 - self.d) / lag
            self.weights[lag] = factor * self.weights[lag - 1]
            self.first[lag] = factor * self.first[lag - 1] - self.weights[lag - 1] / lag
            self.second[lag] = factor * self.second[lag - 1] - 2 * self.first[lag - 1] / lag

    def apply(self, values: np.ndarray, derivative: int = 0) -> np.ndarray:
        """Filter axis 1 independently for each entity and trailing variable."""
        values = np.asarray(values, dtype=float)
        if values.ndim < 2 or values.shape[1] > len(self.weights):
            raise ValueError("values must have a time axis covered by the fractional filter")
        if derivative not in (0, 1, 2):
            raise ValueError("fractional derivative order must be 0, 1, or 2")
        weights = (self.weights, self.first, self.second)[derivative]
        result = weights[0] * values.copy()
        for lag in range(1, values.shape[1]):
            result[:, lag:] += weights[lag] * values[:, :-lag]
        return result

    def forecast(self, filtered_prediction: np.ndarray, residuals: np.ndarray,
                 lengths: np.ndarray) -> np.ndarray:
        """Invert the next filtered residual using each entity's full history."""
        result = np.array(filtered_prediction, dtype=float, copy=True)
        for group, length in enumerate(np.asarray(lengths).reshape(-1)):
            length = int(length)
            if length >= len(self.weights):
                raise ValueError("fractional forecast requires one extra filter coefficient")
            result[group] -= np.sum(
                self.weights[1:length + 1, None] * residuals[group, :length][::-1],
                axis=0,
            )
        return result


class FractionalModel:
    """Shared filter factory for built-in and user-defined likelihood models."""

    @staticmethod
    def fractional_filter(d: float, length: int) -> FractionalFilter:
        return FractionalFilter(d, length)
