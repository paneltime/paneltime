import unittest

import numpy as np
import pandas as pd

import paneltime as pt
from paneltime.main import Input
from paneltime.options import FitOptions
from paneltime.processing.panel import Panel
from paneltime.random_effects.static import (
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

    def test_initializer_excludes_mixed_random_fixed(self):
        from paneltime.processing.initreg import _use_static_start
        data = self._make_data(seed=112)
        for group, time in (("random", "fixed"), ("fixed", "random")):
            with self.subTest(group=group, time=time):
                panel = self._make_panel(data, group_mode=group, time_mode=time)
                self.assertFalse(_use_static_start(panel))

    def test_model_fit_auto_dispatches_closed_form(self):
        data = self._make_data(seed=113)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="fixed", time="none"),
            suppress_output=True,
        )
        self.assertEqual(result.optim_result.its, 0)
        self.assertTrue(result.converged)

    def test_model_fit_auto_dispatches_time_fixed(self):
        data = self._make_data(seed=114)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="fixed", time="fixed"),
            suppress_output=True,
        )
        self.assertEqual(result.optim_result.its, 0)
        self.assertTrue(result.converged)

    def test_model_fit_auto_dispatches_time_random(self):
        data = self._make_time_random_data(seed=214)
        result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
            order=(0, 0, 0),
            garch_order=(0, 0),
            effects=pt.Effects(group="none", time="random"),
            suppress_output=True,
        )
        self.assertGreater(result.optim_result.its, 0)
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
        self.assertGreater(result.optim_result.its, 0)
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

    def test_static_starts_use_shared_likelihood_inference_and_forecasts(self):
        data = self._make_data(seed=116)
        for group, time in (("none", "none"), ("fixed", "none"), ("random", "random")):
            with self.subTest(group=group, time=time):
                result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
                    order=(0, 0, 0), garch_order=(0, 0),
                    effects=pt.Effects(group=group, time=time),
                    optimizer=pt.OptimizerOptions(tolerance=1e-10),
                )
                self.assertTrue(result.converged)
                if group == "random":
                    self.assertGreater(result.optim_result.its, 0)
                else:
                    self.assertEqual(result.optim_result.its, 0)
                self.assertTrue(np.isfinite(result.llf))
                self.assertTrue(np.isfinite(result.bse).all())
                self.assertTrue(np.isfinite(result.robust_bse).all())
                # The variance constant is at its ML value given the estimates.
                omega = result.panel.args.positions["omega"][0]
                scale = max(abs(result.llf), 1.0)
                self.assertLess(abs(result.optim_result.gradient_vector[omega])/scale, 1e-5)
                forecast = result.forecast()
                self.assertGreater(len(forecast), 0)
                self.assertTrue(np.isfinite(forecast).all())

    def test_fixed_effects_intercept_matches_likelihood_maximum(self):
        data = self._make_data(seed=117)
        kwargs = dict(order=(0, 0, 0), garch_order=(0, 0), effects=pt.Effects(group="fixed"))
        model = pt.Model("y ~ x0 + x1", data, entity="id", time="time")
        static = model.fit(**kwargs)
        optimized = model.fit(constraints={}, **kwargs)
        self.assertAlmostEqual(static.params["Intercept"], optimized.params["Intercept"], places=2)
        self.assertAlmostEqual(static.llf, optimized.llf, delta=0.1)

    def test_re_start_is_refined_by_likelihood_optimizer(self):
        from paneltime.maximization.dfpmax import dfpmax
        from paneltime.likelihood.main import LL
        for group, time in (("random", "none"), ("none", "random"), ("random", "random")):
            with self.subTest(group=group, time=time):
                panel = self._make_panel(self._make_data(seed=116), group, time)
                args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
                initial = LL(args, panel)
                result = dfpmax(args, panel)
                self.assertGreater(result["its"], 0)
                self.assertIn(result["conv"], (1, 2))
                self.assertGreaterEqual(result["f"], initial.LL)
                self.assertGreater(np.max(np.abs(result["x"] - args)), 1e-5)
                self.assertNotIn("closed_form", result)

    def test_static_ols_and_one_way_fe_are_likelihood_maxima(self):
        # Unbalanced data: groups start at different dates, so array columns
        # are not dates and time effects must be indexed by date.
        data = self._make_data()
        for group, time in (("none", "none"), ("fixed", "none"), ("none", "fixed"), ("fixed", "fixed")):
            with self.subTest(group=group, time=time):
                result = pt.Model("y ~ x0 + x1", data, entity="id", time="time").fit(
                    order=(0, 0, 0), garch_order=(0, 0),
                    effects=pt.Effects(group=group, time=time),
                )
                beta = list(result.panel.args.positions["beta"])
                np.testing.assert_allclose(result.optim_result.gradient_vector[beta], 0.0, atol=1e-6)

    def test_one_way_between_variance_uses_count_normalization(self):
        from paneltime.random_effects.static import _one_way_variance_component
        X = np.ones((4, 1))
        y = np.array([-1.0, 1.0, -1.0, 1.0])
        for count in (5, 20):
            counts = np.full(4, count, dtype=float)
            actual = _one_way_variance_component(X, y, counts, counts > 0, 0.6)
            self.assertAlmostEqual(actual, np.var(y, ddof=1) - 0.6/count)
        counts = np.array([3., 5., 8., 11.])
        root = np.sqrt(counts)
        M = np.eye(4) - np.outer(root, root)/counts.sum()
        expected = ((root*y) @ M @ (root*y) - 0.6*np.trace(M)) / np.trace(M @ np.diag(counts))
        self.assertAlmostEqual(
            _one_way_variance_component(X, y, counts, counts > 0, 0.6), expected
        )

    def test_effect_reporting_is_independent_of_static_dispatch(self):
        from paneltime.api import PanelEffectResults
        for group, time in (("fixed", "fixed"), ("random", "none"), ("random", "random")):
            with self.subTest(group=group, time=time):
                result = pt.Model("y ~ x0 + x1", self._make_data(), entity="id", time="time").fit(
                    order=(0, 0, 0), garch_order=(0, 0),
                    effects=pt.Effects(group=group, time=time), suppress_output=True,
                )
                reference = PanelEffectResults(result.panel, result.ll)
                for axis in ("group", "time"):
                    actual = getattr(result.effect_results, axis)
                    expected = getattr(reference, axis)
                    self.assertEqual(actual.variance, expected.variance)
                    if actual.estimates is not None:
                        np.testing.assert_allclose(actual.estimates, expected.estimates)

    def test_corrected_effect_likelihood_derivatives(self):
        from paneltime.likelihood.main import LL
        from paneltime.likelihood.calculus import gradient, hessian
        for group, time in (("fixed", "fixed"), ("random", "none"), ("none", "random")):
            with self.subTest(group=group, time=time):
                options = FitOptions(
                    order=(1, 1, 0), garch_order=(1, 1),
                    effects=pt.Effects(group=group, time=time), suppress_output=True,
                ).to_engine_options()
                panel = Panel(Input(self._make_data(), "y ~ x0 + x1", "id", "time", options, None, None), options)
                panel.init()
                args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
                for name, value in (("rho", .2), ("lambda", .1), ("gamma", .2), ("psi", .1), ("omega", .5)):
                    args[list(panel.args.positions[name])] = value

                def evaluate(values):
                    ll = LL(values, panel)
                    g = gradient(panel)
                    score, _ = g.get(ll)
                    return ll, g, score

                ll, g, score = evaluate(args)
                H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
                step = 1e-6
                for j in range(len(args)):
                    plus, minus = args.copy(), args.copy()
                    plus[j] += step
                    minus[j] -= step
                    lp, _, gp = evaluate(plus)
                    lm, _, gm = evaluate(minus)
                    self.assertAlmostEqual(score[j], (lp.LL-lm.LL)/(2*step), delta=1e-4)
                    np.testing.assert_allclose(H[:, j], (gp-gm)/(2*step), rtol=2e-4, atol=1e-4)

    def test_static_time_fixed_effects_match_dummy_regression(self):
        data = self._make_data()
        dummies = [pd.get_dummies(data["time"], drop_first=True, dtype=float)]
        for group, time in (("none", "fixed"), ("fixed", "fixed")):
            if group == "fixed":
                dummies.append(pd.get_dummies(data["id"], drop_first=True, dtype=float))
            X = np.column_stack(
                [np.ones(len(data)), data[["x0", "x1"]].to_numpy()] + [d.to_numpy() for d in dummies]
            )
            expected = np.linalg.lstsq(X, data["y"].to_numpy(), rcond=None)[0][1:3]
            est = fit_group_effects_stationary(self._make_panel(data, group, time))
            np.testing.assert_allclose(est.beta.to_numpy()[1:], expected, atol=1e-8)
            self.assertEqual(len(est.time_effects), data["time"].nunique())


if __name__ == "__main__":
    unittest.main()
