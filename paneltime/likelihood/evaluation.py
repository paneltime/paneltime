"""Evaluate a prepared model without running an optimizer or building results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calculus import gradient, hessian
from .main import LL


@dataclass(frozen=True)
class LikelihoodEvaluation:
    loglike: float
    score: np.ndarray | None = None
    hessian: np.ndarray | None = None


class LikelihoodEvaluator:
    """Reusable likelihood and analytic derivatives in full coefficient coordinates."""

    def __init__(self, panel):
        self._panel = panel
        self.parameter_names = tuple(panel.args.names_v)
        self.parameter_groups = {
            name: tuple(positions) for name, positions in panel.args.positions.items()
        }
        self.initial_parameters = panel.args.args_init.args_v.copy()
        self._parameters = None
        self._likelihood = None
        self._gradient = None
        self._score = None
        self._hessian = None

    def evaluate(self, parameters, derivative_order=2):
        """Return log likelihood, score and Hessian, without fitting any coefficients.

        Static fit constraints are not applied: callers supply the entire vector,
        including any fixed coefficients. Reuse is limited to the last vector.
        """
        if derivative_order not in (0, 1, 2):
            raise ValueError("derivative_order must be 0, 1 or 2")
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != self.initial_parameters.shape or not np.isfinite(parameters).all():
            raise ValueError("parameters must be a finite vector matching parameter_names")
        if self._parameters is None or not np.array_equal(parameters, self._parameters):
            self._set_parameters(parameters)
        if derivative_order >= 1 and self._gradient is None:
            self._gradient = gradient(self._panel)
            self._score, _ = self._gradient.get(self._likelihood)
        if derivative_order >= 2 and self._hessian is None:
            self._hessian = hessian(self._panel, self._gradient).get(
                self._likelihood, *self._likelihood.llfunc.hessian())
        return LikelihoodEvaluation(
            float(self._likelihood.LL),
            None if derivative_order < 1 else self._score.copy(),
            None if derivative_order < 2 else self._hessian.copy(),
        )

    def _set_parameters(self, parameters):
        """Replace cached work only after a new parameter vector is valid."""
        likelihood = LL(parameters.copy(), self._panel)
        if likelihood.LL is None or not np.isfinite(likelihood.LL):
            raise ValueError(f"Invalid likelihood parameters: {likelihood.err_msg}")
        self._parameters = parameters.copy()
        self._likelihood = likelihood
        self._gradient = None
        self._score = self._hessian = None
