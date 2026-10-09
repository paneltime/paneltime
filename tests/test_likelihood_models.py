import unittest

import numpy as np
import pandas as pd

import paneltime as pt
from paneltime.likelihood.models import Exponential, Hyperbolic, LikelihoodModel, Normal


class NumericNormal(LikelihoodModel):
    def variance_bounds(self, init_var):
        self.minvar, self.maxvar = 1e-30, 1e30
        self.var = np.clip(init_var, self.minvar, self.maxvar)
        self.var_pos = (init_var >= self.minvar) & (init_var <= self.maxvar)

    def variance_definitions(self):
        self.v = self.var
        self.v_inv = 1 / self.var
        self.e2 = self.e**2 + 1e-8

    def loglike(self):
        return -0.5 * (np.log(self.v) + self.e2 / self.v) - 0.5 * np.log(2 * np.pi)


class LikelihoodModelTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([[[0.35]], [[0.8]], [[1.4]]])
        self.variance = np.array([[[0.7]], [[1.5]], [[2.2]]])

    def _finite_differences(self, model_type, variance_arg, a=0.0, k=0.0):
        step = 1e-4
        score_e = np.zeros_like(self.e)
        score_var = np.zeros_like(variance_arg)
        hessian_ee = np.zeros_like(self.e)
        hessian_var_e = np.zeros_like(self.e)
        hessian_var_var = np.zeros_like(variance_arg)

        def evaluate(e, var):
            return model_type(e, var, a, k, None).ll()

        for index in np.ndindex(self.e.shape):
            e_plus = self.e.copy()
            e_minus = self.e.copy()
            e_plus[index] += step
            e_minus[index] -= step
            score_e[index] = (
                evaluate(e_plus, variance_arg)[index]
                - evaluate(e_minus, variance_arg)[index]
            ) / (2 * step)
            hessian_ee[index] = (
                evaluate(e_plus, variance_arg)[index]
                - 2 * evaluate(self.e, variance_arg)[index]
                + evaluate(e_minus, variance_arg)[index]
            ) / step**2

            var_plus = variance_arg.copy()
            var_minus = variance_arg.copy()
            var_plus[index] += step
            var_minus[index] -= step
            score_var[index] = (
                evaluate(self.e, var_plus)[index]
                - evaluate(self.e, var_minus)[index]
            ) / (2 * step)
            hessian_var_var[index] = (
                evaluate(self.e, var_plus)[index]
                - 2 * evaluate(self.e, variance_arg)[index]
                + evaluate(self.e, var_minus)[index]
            ) / step**2
            hessian_var_e[index] = (
                evaluate(e_plus, var_plus)[index]
                - evaluate(e_plus, var_minus)[index]
                - evaluate(e_minus, var_plus)[index]
                + evaluate(e_minus, var_minus)[index]
            ) / (4 * step**2)

        return score_var, score_e, hessian_ee, hessian_var_e, hessian_var_var

    def test_builtin_scores_and_hessians_match_finite_differences(self):
        cases = (
            ("Normal", Normal, self.variance, 0.0, 0.0),
            ("Exponential", Exponential, np.log(self.variance), 0.0, 0.0),
            ("Hyperbolic", Hyperbolic, self.variance, 0.4, 0.2),
        )
        for name, model_type, variance_arg, a, k in cases:
            with self.subTest(model=name):
                model = model_type(self.e.copy(), variance_arg.copy(), a, k, None)
                actual = (*model.dll(), *model.ddll())
                expected = self._finite_differences(model_type, variance_arg, a, k)
                for actual_value, expected_value in zip(actual, expected):
                    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-5, atol=2e-7)

    def test_hyperbolic_inverse_variance_uses_clipped_variance(self):
        model = Hyperbolic(self.e.copy(), np.array([[[1e-40]], [[1.5]], [[1e40]]]), 0.4, 0.2, None)
        np.testing.assert_allclose(model.v_inv, 1 / model.var)

    def test_custom_likelihood_fallback_matches_normal_derivatives_and_restores_state(self):
        model = NumericNormal(self.e.copy(), self.variance.copy())
        reference = Normal(self.e.copy(), self.variance.copy(), 0.0, 0.0, None)
        score = model.score()
        hessian = model.hessian()
        self.assertEqual(len(score), 2)
        self.assertEqual(len(hessian), 3)
        for actual, expected in zip(score, reference.dll()):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
        for actual, expected in zip(hessian, reference.ddll()):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
        np.testing.assert_array_equal(model.e, self.e)
        np.testing.assert_array_equal(model.var, self.variance)

    def test_custom_likelihood_supports_scalar_initialization_probe(self):
        model = NumericNormal(-10, 10)
        self.assertTrue(np.isscalar(model.h_val))
        self.assertTrue(np.isscalar(model.h_2e_val))

    def test_custom_likelihood_fits_through_paneltime(self):
        rng = np.random.default_rng(7)
        groups = np.repeat(np.arange(4), 30)
        dates = np.tile(np.arange(30), 4)
        x = rng.normal(size=len(groups))
        y = 0.4 + 1.2 * x + rng.normal(size=len(groups))
        data = pd.DataFrame({"IDs": groups, "dates": dates, "X0": x, "Y": y})

        result = pt.Model("Y ~ X0", data, entity="IDs", time="dates").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            likelihood=NumericNormal,
            suppress_output=True,
        )

        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.params.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()