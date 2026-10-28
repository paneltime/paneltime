import unittest

import numpy as np
import pandas as pd

import paneltime as pt
from paneltime.main import Input
from paneltime.options import FitOptions
from paneltime.processing.panel import Panel
from paneltime.random_effects.static import (
    StationaryEffectsError,
    fit_group_effects_stationary,
)


class StationaryRandomEffectsTests(unittest.TestCase):
    @staticmethod
    def _make_data(seed=123, n=18, t=9):
        rng = np.random.default_rng(seed)
        rows = []
        alpha = rng.normal(scale=0.8, size=n)
        beta = np.array([0.5, -0.7])
        for i in range(n):
            for tt in range(t):
                # Mild unbalance to verify masked observations are handled.
                if (i + tt) % 11 == 0:
                    continue
                x0 = rng.normal()
                x1 = rng.normal()
                eps = rng.normal(scale=0.6)
                y = 1.2 + beta[0] * x0 + beta[1] * x1 + alpha[i] + eps
                rows.append((i, tt, x0, x1, y))
        return pd.DataFrame(rows, columns=["id", "time", "x0", "x1", "y"])

    @staticmethod
    def _make_time_random_data(seed=321, n=18, t=9):
        rng = np.random.default_rng(seed)
        rows = []
        lambda_t = rng.normal(scale=0.8, size=t)
        beta = np.array([0.5, -0.7])
        for i in range(n):
            for tt in range(t):
                # Mild unbalance to verify masked observations are handled.
                if (2 * i + tt) % 13 == 0:
                    continue
                x0 = rng.normal()
                x1 = rng.normal()
                eps = rng.normal(scale=0.6)
                y = 1.2 + beta[0] * x0 + beta[1] * x1 + lambda_t[tt] + eps
                rows.append((i, tt, x0, x1, y))
        return pd.DataFrame(rows, columns=["id", "time", "x0", "x1", "y"])

    @staticmethod
    def _make_panel(data, group_mode="fixed", time_mode="none"):
        options = FitOptions(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group=group_mode, time=time_mode),
            suppress_output=True,
        ).to_engine_options()
        panel = Panel(Input(data, "y ~ x0 + x1", "id", "time", options, None, None), options)
        panel.init()
        return panel

    def _compare_betas(self, result, closed_form):
        expected = result.params[result.names_independents]
        got = closed_form.beta[result.names_independents]
        return expected, got

    def test_fixed_effects_closed_form_matches_fit(self):
        data = self._make_data(seed=110)
        model_result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="fixed", time="none"),
            suppress_output=True,
        )
        panel = self._make_panel(data, group_mode="fixed", time_mode="none")
        closed_form = fit_group_effects_stationary(panel, mode="fixed")

        expected, got = self._compare_betas(model_result, closed_form)
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-3, atol=1e-3)

    def test_random_effects_closed_form_close_to_fit(self):
        data = self._make_data(seed=111)
        model_result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="random", time="none"),
            suppress_output=True,
        )
        panel = self._make_panel(data, group_mode="random", time_mode="none")
        closed_form = fit_group_effects_stationary(panel, mode="random")

        expected, got = self._compare_betas(model_result, closed_form)
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=7e-2, atol=7e-2)
        self.assertGreaterEqual(closed_form.sigma_alpha2, 0.0)

    def test_random_time_effects_closed_form_close_to_fit(self):
        data = self._make_time_random_data(seed=211)
        model_result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="none", time="random"),
            suppress_output=True,
        )
        panel = self._make_panel(data, group_mode="none", time_mode="random")
        closed_form = fit_group_effects_stationary(panel, mode="none", time_mode="random")

        expected, got = self._compare_betas(model_result, closed_form)
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=7e-2, atol=7e-2)
        self.assertGreaterEqual(closed_form.sigma_lambda2, 0.0)

    def test_rejects_mixed_random_fixed(self):
        data = self._make_data(seed=112)
        panel = self._make_panel(data, group_mode="fixed", time_mode="fixed")
        with self.assertRaises(StationaryEffectsError):
            fit_group_effects_stationary(panel, mode="random", time_mode="fixed")

    def test_model_fit_auto_dispatches_closed_form(self):
        data = self._make_data(seed=113)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="fixed", time="none"),
            suppress_output=True,
        )
        self.assertEqual(result.method, "Closed-form stationary (group fixed/time none)")
        self.assertTrue(result.converged)

    def test_model_fit_auto_dispatches_time_fixed(self):
        data = self._make_data(seed=114)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="fixed", time="fixed"),
            suppress_output=True,
        )
        self.assertEqual(result.method, "Closed-form stationary (group fixed/time fixed)")
        self.assertTrue(result.converged)

    def test_model_fit_auto_dispatches_time_random(self):
        data = self._make_time_random_data(seed=214)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="none", time="random"),
            suppress_output=True,
        )
        self.assertEqual(result.method, "Closed-form stationary (group none/time random)")
        self.assertTrue(result.converged)

    def test_random_two_way_effects_closed_form_close_to_fit(self):
        data = self._make_data(seed=311)
        model_result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="random", time="random"),
            suppress_output=True,
        )
        panel = self._make_panel(data, group_mode="random", time_mode="random")
        closed_form = fit_group_effects_stationary(panel, mode="random", time_mode="random")

        expected, got = self._compare_betas(model_result, closed_form)
        np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1.2e-1, atol=1.2e-1)
        self.assertGreaterEqual(closed_form.sigma_alpha2, 0.0)
        self.assertGreaterEqual(closed_form.sigma_lambda2, 0.0)

    def test_model_fit_auto_dispatches_two_way_random(self):
        data = self._make_data(seed=312)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="random", time="random"),
            suppress_output=True,
        )
        self.assertEqual(result.method, "Closed-form stationary (group random/time random)")
        self.assertTrue(result.converged)

    def test_model_fit_falls_back_for_mixed_random_fixed(self):
        data = self._make_data(seed=115)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="random", time="fixed"),
            suppress_output=True,
        )
        self.assertFalse(hasattr(result, "method"))
        self.assertTrue(hasattr(result, "llf"))


if __name__ == "__main__":
    unittest.main()
