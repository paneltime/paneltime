import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import paneltime as pt
from paneltime import cfunctions
from paneltime.main import Input
from paneltime.options import Effects, FitOptions
from paneltime.processing import arma_start, initreg
from paneltime.processing.panel import Panel


class ARMAStartTests(unittest.TestCase):
    def working_panel(self, x, y, included=None, p=2, q=2, d=0):
        if included is None:
            included = np.ones(y.shape, dtype=bool)
        return SimpleNamespace(
            X=x, Y=y[:, :, None], pqdkm=(p, q, d, 0, 0),
            included={3: included[:, :, None]}, NT=included.sum(),
            T_i=included.sum(axis=1)[:, None, None],
            date_count_mtrx=None, date_count=None,
            options=SimpleNamespace(fixed_random_group_eff=0, fixed_random_time_eff=0),
        )

    def make_data(self, n=3, t=160):
        rng = np.random.default_rng(301)
        x = np.stack((np.ones((n, t)), rng.normal(size=(n, t))), -1)
        shocks = rng.normal(size=(n, t))
        u = shocks.copy()
        for j in range(2, t):
            u[:, j] += (0.5*u[:, j-1] - 0.2*u[:, j-2]
                        + 0.3*shocks[:, j-1] + 0.2*shocks[:, j-2])
        return x, 0.4 + 1.2*x[:, :, 1] + u

    def test_reflection_roots_and_jacobian(self):
        for size in (0, 1, 2, 4):
            eta = np.linspace(-0.7, 0.9, size)
            coefficients, derivative = arma_start.reflection_coefficients(eta)
            if size:
                self.assertLess(np.abs(np.roots(np.r_[1., -coefficients])).max(), 1)
            np.testing.assert_allclose(
                arma_start.reflection_coefficients(
                    arma_start.reflection_coordinates(coefficients),
                )[0], coefficients, atol=1e-12,
            )
            for j in range(size):
                step = np.eye(size)[j] * 1e-6
                numeric = (
                    arma_start.reflection_coefficients(eta+step)[0]
                    - arma_start.reflection_coefficients(eta-step)[0]
                ) / 2e-6
                np.testing.assert_allclose(derivative[:, j], numeric, atol=1e-9)
        eta = arma_start.reflection_coordinates(np.array([1.4, 0.5]))
        coefficients = arma_start.reflection_coefficients(eta)[0]
        self.assertLess(np.abs(np.roots(np.r_[1., -coefficients])).max(), 1)

    def test_css_jacobian_and_independent_runs(self):
        x, y = self.make_data(n=2, t=18)
        included = np.ones(y.shape, dtype=bool)
        included[0, 7] = False
        included[1, 13:] = False
        lengths = arma_start.run_lengths(included)
        parameters = np.array([0.4, 1.2, 0.3, -0.2, -0.3, 0.1])
        e, derivative = arma_start.css_residuals(parameters, x, y, 2, 2, lengths)
        for j in range(len(parameters)):
            step = np.eye(len(parameters))[j] * 1e-6
            numeric = (
                arma_start.css_residuals(parameters+step, x, y, 2, 2, lengths, False)[0]
                - arma_start.css_residuals(parameters-step, x, y, 2, 2, lengths, False)[0]
            ) / 2e-6
            np.testing.assert_allclose(derivative[:, :, j], numeric, atol=1e-8)
        ar = arma_start.reflection_coefficients(parameters[2:4])[0]
        ma = -arma_start.reflection_coefficients(parameters[4:])[0]
        for i, start, end in ((0, 0, 7), (0, 8, 18), (1, 0, 13)):
            residual = y[i, start:end] - x[i, start:end] @ parameters[:2]
            expected = residual.copy()
            for t in range(end-start):
                for j in range(min(2, t)):
                    expected[t] -= ar[j]*residual[t-j-1] + ma[j]*expected[t-j-1]
            np.testing.assert_allclose(e[i, start:end], expected, atol=1e-12)
        np.testing.assert_array_equal(e[~included], 0)

    def test_native_filter_matches_scalar_recursion_across_runs_and_channels(self):
        rng = np.random.default_rng(912)
        values = rng.normal(size=(4, 80, 7))
        included = np.ones(values.shape[:2], dtype=bool)
        included[0, :3] = False
        included[0, 30:33] = False
        included[1, 55:] = False
        included[2] = False
        included[3, 1::2] = False
        lengths = arma_start.run_lengths(included)
        before = values.copy()
        for ma in (np.empty(0), np.array([0.3, 0.2]), np.array([-0.98]),
                   np.array([0.2, -0.1, 0.05])):
            expected = values * included[:, :, None]
            for t in range(values.shape[1]):
                for j in range(min(len(ma), t)):
                    expected[:, t] -= (
                        ma[j] * expected[:, t-j-1] * (lengths[:, t] > j+1)[:, None]
                    )
            with patch.object(cfunctions, 'armas', wraps=cfunctions.armas) as native:
                actual = arma_start.inverse_ma_filter(values, ma, lengths)
            self.assertEqual(native.call_count, int(bool(len(ma))))
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            np.testing.assert_array_equal(actual[~included], 0)
        np.testing.assert_array_equal(values, before)
        with patch.object(cfunctions, 'armas', side_effect=AssertionError('Native called')):
            result = arma_start.inverse_ma_filter(values, np.array([0.3]),
                                                   np.zeros_like(lengths))
        np.testing.assert_array_equal(result, 0)
        noncontiguous = values[:, :, ::2]
        result = arma_start.inverse_ma_filter(noncontiguous, np.array([0.3]), lengths)
        contiguous = arma_start.inverse_ma_filter(
            np.ascontiguousarray(noncontiguous), np.array([0.3]), lengths,
        )
        np.testing.assert_array_equal(result, contiguous)

    def test_native_status_failure_is_not_silently_ignored(self):
        with patch.object(cfunctions.cfunct, 'armas', return_value=-1):
            with self.assertRaisesRegex(RuntimeError, 'failed with status -1'):
                arma_start.inverse_ma_filter(
                    np.ones((1, 5, 1)), np.array([0.3]),
                    arma_start.run_lengths(np.ones((1, 5), dtype=bool)),
                )

    def test_full_order_fit_improves_css_and_respects_scaling(self):
        x, y = self.make_data()
        included = np.ones(y.shape, dtype=bool)
        beta = np.linalg.lstsq(x.reshape(-1, 2), y.ravel(), rcond=None)[0][:, None]
        fitted = arma_start.fit_arma(self.working_panel(x, y, included))
        beta_fit, ar, ma, e = fitted
        baseline = (y - x @ beta[:, 0])[:, 2:]
        self.assertLess(np.mean(e[:, 2:]**2), 0.85*np.mean(baseline**2))
        self.assertGreater(abs(ar[1]), 0.01)
        self.assertGreater(abs(ma[1]), 0.01)
        for coefficients in (ar, -ma):
            self.assertLess(np.abs(np.roots(np.r_[1., -coefficients])).max(), 1)
        scaled_x = x * np.array([1., 1000.])
        scaled = arma_start.fit_arma(
            self.working_panel(scaled_x, 10*y, included),
        )
        np.testing.assert_allclose(scaled[0]*np.array([[1.], [1000.]])/10,
                                   beta_fit, atol=1e-8)
        np.testing.assert_allclose(scaled[1], ar, atol=1e-8)
        np.testing.assert_allclose(scaled[2], ma, atol=1e-8)
        np.testing.assert_allclose(scaled[3]/10, e, atol=1e-8)

    def test_refinement_has_a_short_iteration_budget(self):
        x, y = self.make_data()
        with patch.object(arma_start, 'css_residuals',
                          wraps=arma_start.css_residuals) as evaluate:
            arma_start.fit_arma(self.working_panel(x, y))
        derivative_calls = sum(
            call.kwargs.get('jacobian', True) for call in evaluate.call_args_list
        )
        self.assertLessEqual(derivative_calls, 11)

    def test_static_initialization_retains_ols_and_moment_rule(self):
        panel = self.make_panel(order=(0, 0, 0))
        beta, residuals = initreg.stat.OLS(panel, panel.X, panel.Y, return_e=True)
        correlations = initreg.stat.correlogram(panel, residuals, 2, center=True)[1:]
        squared = initreg.stat.correlogram(
            panel, residuals**2*panel.included[3], 2, center=True,
        )[1:] - correlations**2
        arch, garch = initreg.GARCH_process_calc(squared, *panel.pqdkm[3:])
        variance = panel.var(residuals)
        with patch.object(arma_start, 'fit_arma', side_effect=AssertionError('CSS called')):
            result = initreg.start_values(panel)
        np.testing.assert_allclose(result[0], beta)
        self.assertEqual(result[1].size, 0)
        self.assertEqual(result[2].size, 0)
        np.testing.assert_allclose(result[3:6], [arch, garch, variance])
        self.assertAlmostEqual(result[7], panel.h_func(0, variance, variance)*(1-arch-garch))

    def test_ar_only_ma_only_and_short_data(self):
        x, y = self.make_data(n=1, t=160)
        for p, q in ((2, 0), (0, 2), (1, 1)):
            with warnings.catch_warnings(record=True):
                fitted = arma_start.fit_arma(self.working_panel(x, y, p=p, q=q))
            self.assertEqual(fitted[1].shape, (p,))
            self.assertEqual(fitted[2].shape, (q,))
            self.assertTrue(all(np.isfinite(value).all() for value in fitted))
        with self.assertWarnsRegex(RuntimeWarning, 'Too few conditional'):
            fitted = arma_start.fit_arma(self.working_panel(x[:, :4], y[:, :4]))
        np.testing.assert_array_equal(fitted[1], 0)
        with self.assertWarnsRegex(RuntimeWarning, 'underidentified'):
            arma_start.auxiliary_start(np.zeros((1, 20)), 2, 2,
                                       arma_start.run_lengths(np.ones((1, 20), bool)))

    def test_panel_metadata_controls_mask_and_orders_without_mutation(self):
        x, y = self.make_data(n=2, t=24)
        included = np.ones(y.shape, dtype=bool)
        included[0, 8] = False
        included[1, 20:] = False
        panel = self.working_panel(x, y, included, p=1, q=2, d=1)
        expected = included.copy()
        expected[:, 0] = False
        np.testing.assert_array_equal(arma_start.observed_mask(panel), expected)
        x_before, y_before = x.copy(), y.copy()
        with warnings.catch_warnings(record=True):
            result = arma_start.fit_arma(panel)
        self.assertEqual(result[1].shape, (1,))
        self.assertEqual(result[2].shape, (2,))
        np.testing.assert_array_equal(result[3][~expected], 0)
        np.testing.assert_array_equal(panel.included[3][:, :, 0], included)
        np.testing.assert_array_equal(panel.X, x_before)
        np.testing.assert_array_equal(panel.Y[:, :, 0], y_before)

    def make_panel(self, order=(2, 2, 0), effects=None, **kwargs):
        x, y = self.make_data(n=4, t=40)
        data = pd.DataFrame({
            'id': np.repeat(np.arange(4), 40), 'time': np.tile(np.arange(40), 4),
            'x': x[:, :, 1].ravel(), 'y': y.ravel(),
        })
        # Exercise trailing padding in the panel representation.
        data = data.drop(data[(data.id == 3) & (data.time >= 32)].index)
        options = FitOptions(order=order, garch_order=(2, 2),
                             effects=effects or Effects(), **kwargs).to_engine_options()
        panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
        with warnings.catch_warnings(record=True):
            panel.init()
        return panel

    def test_panel_wiring_effects_differencing_and_variance(self):
        for order, effects in (
            ((2, 2, 0), Effects()),
            ((1, 1, 1), Effects(group='fixed', time='random')),
            ((2, 1, 0), Effects(group='random', time='fixed')),
        ):
            panel = self.make_panel(order=order, effects=effects)
            args = panel.args.args_init.args_d
            self.assertTrue(np.isfinite(panel.args.args_init.args_v).all())
            self.assertEqual(args['rho'].shape, (order[0], 1))
            self.assertEqual(args['lambda'].shape, (order[1], 1))
            if order[0] == 2:
                self.assertNotEqual(args['rho'][1, 0], 0)
            self.assertGreater(panel.args.init_var, 0)
            persistence = args['psi'].sum() + args['gamma'].sum()
            self.assertAlmostEqual(args['omega'][0, 0],
                                   panel.h_func(0, panel.args.init_var, panel.args.init_var)
                                   * (1-persistence))
        panel = self.make_panel(order=(0, 0, 0))
        self.assertTrue(np.isfinite(panel.args.args_init.args_v).all())

    def test_explicit_start_skips_css_and_remains_unchanged(self):
        panel = self.make_panel()
        start = panel.args.args_init.args_d
        panel.options.arguments = start
        with patch.object(initreg, 'start_values',
                          side_effect=AssertionError('Automatic initialization called')):
            panel.args.set_init_args(panel)
        self.assertTrue(panel.args.initial_user_defined)
        for name, value in start.items():
            np.testing.assert_array_equal(panel.args.args_init.args_d[name], value)

    def test_public_fit_reports_initialization_warning_without_aborting(self):
        x, y = self.make_data(n=1, t=100)
        data = pd.DataFrame({'x': x[0, :, 1], 'y': y[0],
                             'time': np.arange(100), 'id': 0})
        fit_arma = arma_start.fit_arma

        def unfinished_start(*args, **kwargs):
            result = fit_arma(*args, **kwargs)
            warnings.warn('Preliminary iteration limit',
                          arma_start.ARMAInitializationWarning)
            return result

        with warnings.catch_warnings(record=True) as caught:
            with patch.object(arma_start, 'fit_arma', side_effect=unfinished_start):
                result = pt.Model('y ~ x', data, entity='id', time='time').fit(
                                      order=(1, 1, 0), garch_order=(0, 0))
        self.assertTrue(np.isfinite(result.llf))
        self.assertTrue(any(
            warning.category is arma_start.ARMAInitializationWarning
            and str(warning.message) == 'Preliminary iteration limit'
            for warning in caught
        ))

        def numerical_error(*args, **kwargs):
            warnings.warn('Unexpected numerical failure', RuntimeWarning)

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with patch.object(arma_start, 'fit_arma', side_effect=numerical_error):
                with self.assertRaisesRegex(RuntimeWarning, 'Unexpected numerical failure'):
                    pt.Model('y ~ x', data, entity='id', time='time').fit(
                        order=(1, 1, 0), garch_order=(0, 0))


if __name__ == '__main__':
    unittest.main()
