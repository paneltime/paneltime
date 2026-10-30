import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

import paneltime as pt
import paneltime.api as api
from paneltime.cfunctions import _library_name
from paneltime.likelihood.calculus import gradient, hessian
from paneltime.likelihood.main import LL
from paneltime.likelihood.symbolic import build_h_function
from paneltime.options import FitOptions
from paneltime.processing.model_parser import get_names, pool_func
from paneltime.random_effects import REObj
from paneltime.output.output import sandwich
from paneltime.maximization.direction import get as get_direction
from paneltime.main import Input
from paneltime.main import execute
from paneltime.processing.panel import Panel

try:
    import sympy as sp
except ImportError:
    sp = None


class PackageWorkflowTests(unittest.TestCase):
    def test_cached_date_means_match_reference_for_ragged_and_balanced_panels(self):
        from paneltime.random_effects.dynamic import mean_time, mean_time_slow
        rng = np.random.default_rng(107)
        for ragged in (False, True):
            data = pd.DataFrame({
                'id': np.repeat(np.arange(5), 9),
                'time': np.tile(np.arange(9), 5),
                'y': rng.normal(size=45), 'x': rng.normal(size=45),
            })
            if ragged:
                data = data[~((data.id < 2) & (data.time < 3))]
            options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
            panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
            panel.init()
            for trailing in ((1,), (3,), (2, 3)):
                with self.subTest(ragged=ragged, trailing=trailing):
                    base = rng.normal(size=panel.Y.shape[:2] + trailing)
                    for values in (base, np.asfortranarray(base), base[:, ::-1]):
                        original = values.copy()
                        actual = mean_time(panel, values)
                        expected = mean_time_slow(panel, values)
                        for a, e in zip(actual, expected):
                            np.testing.assert_allclose(a, e, atol=1e-14)
                        np.testing.assert_allclose(
                            mean_time(panel, values, True), mean_time_slow(panel, values, True), atol=1e-14,
                        )
                        np.testing.assert_array_equal(values, original)

    def test_sandwich_pair_reuses_only_identical_parameter_sets(self):
        from paneltime.output import output
        rng = np.random.default_rng(108)
        H = -np.eye(3)
        G = rng.normal(size=(1, 8, 3))
        g = np.zeros(3)
        panel = SimpleNamespace(
            lost_obs=0, NT=8, N=1, n_dates=8, X=np.ones((1, 8, 3)),
            input=SimpleNamespace(has_intercept=True),
            options=SimpleNamespace(fixed_random_group_eff=0, fixed_random_time_eff=0),
        )
        for fixed in ({}, {0: SimpleNamespace(assco_ix=None)}, {0: SimpleNamespace(assco_ix=1)}):
            constr = SimpleNamespace(fixed=fixed)
            expected = (sandwich(H, G, g, constr, panel, 0),
                        sandwich(H, G, g, constr, panel, 0, oposite=True))
            with patch.object(output, 'sandwich', wraps=sandwich) as calculation:
                actual = output.sandwich_pair(H, G, g, constr, panel, 0)
            self.assertEqual(calculation.call_count, 2 if fixed else 1)
            for a_block, e_block in zip(actual, expected):
                for a, e in zip(a_block, e_block):
                    np.testing.assert_allclose(a, e, equal_nan=True)
            self.assertFalse(np.shares_memory(actual[0][0], actual[1][0]))

    def test_reporting_reuses_final_effect_calculations(self):
        from paneltime.api import PanelEffectResults
        from paneltime.output.output import RandomEffects, grand_mean_variance
        panel = self._presample_panel((0, 0, 0), (0, 0), n=10, t=15)
        panel.options.fixed_random_group_eff = 2
        panel.options.fixed_random_time_eff = 2
        rng = np.random.default_rng(109)
        panel.Y += rng.normal(size=(10, 1, 1)) + rng.normal(size=(1, 15, 1))
        ll = LL(panel.args.args_init.args_v, panel)
        fresh_group = REObj(panel, True, panel.T_i, panel.T_i, 2)
        fresh_time = REObj(panel, False, panel.date_count_mtrx, panel.date_count, 2)
        group_adjustment = fresh_group.RE(ll.e, panel)
        time_adjustment = fresh_time.RE(ll.e, panel)
        expected_variance = fresh_group.v_var/panel.N + fresh_time.v_var/panel.n_dates
        with patch.object(REObj, 'RE', side_effect=AssertionError('Repeated effect estimation')):
            effects = PanelEffectResults(panel, ll)
            legacy = RandomEffects(panel, ll)
            variance = grand_mean_variance(panel, ll)
        np.testing.assert_allclose(legacy.residuals_i, group_adjustment)
        np.testing.assert_allclose(legacy.residuals_t, time_adjustment)
        self.assertAlmostEqual(effects.group.variance, fresh_group.v_var)
        self.assertAlmostEqual(effects.time.variance, fresh_time.v_var)
        self.assertAlmostEqual(variance, expected_variance)

    @staticmethod
    def _presample_panel(order, garch, seed=17, n=5, t=12):
        rng = np.random.default_rng(seed)
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'x': rng.normal(size=n*t), 'y': rng.normal(size=n*t),
        })
        options = FitOptions(
            order=order, garch_order=garch, suppress_output=True,
        ).to_engine_options()
        panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
        panel.init()
        return panel

    def test_presample_variance_is_sample_variance(self):
        panel = self._presample_panel((1, 0, 1), (1, 1))
        positions = panel.args.positions
        args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        args[list(positions['omega'])] = 0.6
        args[list(positions['gamma'])] = 0.3
        args[list(positions['psi'])] = 0.1
        ll = LL(args, panel)
        np.testing.assert_allclose(ll.presample_var, panel.args.init_var)
        self.assertNotIn('initvar', panel.args.names_v)

        args[list(positions['omega'])] = 0.2
        np.testing.assert_allclose(LL(args, panel).presample_var, panel.args.init_var)
        args[list(positions['gamma'])] = 0.95
        self.assertIsNotNone(LL(args, panel).LL)

    def test_likelihood_has_no_presample_effect_when_garch_is_zero(self):
        for order, garch in (((0, 0, 0), (0, 0)), ((1, 0, 1), (0, 0)), ((0, 0, 0), (1, 1))):
            panel = self._presample_panel(order, garch, seed=5, n=4, t=10)
            args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
            for name in ('gamma', 'psi'):
                args[list(panel.args.positions.get(name, []))] = 0.0
            args[list(panel.args.positions['omega'])] = 0.7
            ll = LL(args, panel)
            incl = panel.included[3] > 0
            np.testing.assert_allclose(ll.llfunc.var[incl], 0.7)

    def test_presample_lags_follow_variance_recursion(self):
        panel = self._presample_panel((0, 0, 0), (2, 2), n=4, t=10)
        positions = panel.args.positions
        args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        gamma, psi = [0.3, 0.1], [0.1, 0.05]
        args[list(positions['gamma'])] = gamma
        args[list(positions['psi'])] = psi
        args[list(positions['omega'])] = 0.5
        ll = LL(args, panel)
        expected = np.full_like(ll.var, 0.5)
        for t in range(ll.var.shape[1]):
            for j in range(1, 3):
                prior_var = expected[:, t-j, 0] if t >= j else panel.args.init_var
                prior_h = ll.h_val[:, t-j, 0] if t >= j else panel.args.init_var
                expected[:, t, 0] += gamma[j-1]*prior_var + psi[j-1]*prior_h
        np.testing.assert_allclose(ll.var, expected)
        self.assertAlmostEqual(ll.LL, np.sum(ll.ll_value * panel.included[3]))

    def test_presample_derivatives_match_finite_differences(self):
        for garch in ((1, 1), (2, 2)):
            with self.subTest(garch=garch):
                self._check_presample_derivatives(garch)

    def _check_presample_derivatives(self, garch):
        panel = self._presample_panel((1, 0, 1), garch, seed=3, n=4, t=10)
        positions = panel.args.positions
        args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        args[list(positions['gamma'])] = 0.15
        args[list(positions['psi'])] = 0.15
        args[list(positions['omega'])] = 0.5

        def derivatives(a):
            ll = LL(a, panel)
            g = gradient(panel)
            score, _ = g.get(ll)
            return ll, g, score

        ll, g, score = derivatives(args)
        H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
        h = 1e-6
        for i in range(len(args)):
            up, down = args.copy(), args.copy()
            up[i] += h
            down[i] -= h
            self.assertAlmostEqual(
                score[i], (LL(up, panel).LL - LL(down, panel).LL)/(2*h), delta=1e-5,
            )
            np.testing.assert_allclose(
                H[:, i], (derivatives(up)[2] - derivatives(down)[2])/(2*h), atol=1e-4,
            )

    def test_dynamic_time_effects_are_indexed_by_date(self):
        rng = np.random.default_rng(8)
        n, t = 12, 10
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'x': rng.normal(size=n*t),
        })
        true_effects = 2*rng.normal(size=t)
        data['y'] = data.x + true_effects[data.time] + 0.3*rng.normal(size=n*t)
        data = data[~((data.id < 4) & (data.time < 3))]
        result = pt.Model('y ~ x', data, entity='id', time='time').fit(
            order=(0, 0, 0), garch_order=(1, 1), effects=pt.Effects(time='fixed'),
            suppress_output=True,
        )
        estimates = result.effect_results.time.estimates
        np.testing.assert_array_equal(estimates.index, np.arange(t))
        self.assertGreater(np.corrcoef(estimates, true_effects)[0, 1], 0.99)

    @staticmethod
    def _arma_panel(rho, lmbda, seed):
        rng = np.random.default_rng(seed)
        n, t = 50, 40
        e = rng.normal(size=(n, t))
        u = np.zeros((n, t))
        for j in range(t):
            u[:, j] = e[:, j]
            if j > 0:
                u[:, j] += rho*u[:, j-1] + lmbda*e[:, j-1]
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'x': rng.normal(size=n*t),
        })
        data['y'] = 1 + 0.5*data.x + u.ravel()
        return data

    def test_arma_common_factor_is_removed_on_white_noise(self):
        for seed, garch in [(1, (0, 0)), (2, (0, 0)), (2, (1, 1))]:
            with self.subTest(seed=seed, garch=garch):
                data = self._arma_panel(0.0, 0.0, seed)
                model = pt.Model('y ~ x', data, entity='id', time='time')
                result = model.fit(order=(1, 1, 0), garch_order=garch, suppress_output=True)
                restricted = model.fit(
                    order=(1, 1, 0), garch_order=garch, constraints={'rho': 0.0, 'lambda': 0.0},
                    suppress_output=True,
                )
                self.assertEqual(result.params.iloc[2], 0.0)
                self.assertEqual(result.params.iloc[3], 0.0)
                self.assertIn('common factor', result.optim_result.msg)
                # GARCH is not identified on iid variance, so the two fits may stop
                # at slightly different points of a flat likelihood.
                self.assertAlmostEqual(result.optim_result.ll.LL, restricted.optim_result.ll.LL, delta=0.1)
                self.assertTrue(np.all(np.isfinite(result.pvalues.iloc[:2])))

    def test_genuine_arma_is_not_reduced(self):
        for garch in [(0, 0), (1, 1)]:
            with self.subTest(garch=garch):
                data = self._arma_panel(0.5, -0.3, 3)
                result = pt.Model('y ~ x', data, entity='id', time='time').fit(
                    order=(1, 1, 0), garch_order=garch, suppress_output=True,
                )
                self.assertNotIn('common factor', result.optim_result.msg)
                self.assertAlmostEqual(result.params.iloc[2], 0.5, delta=0.15)
                self.assertAlmostEqual(result.params.iloc[3], -0.3, delta=0.15)

    def test_unavailable_covariance_always_returns_four_outputs(self):
        panel = SimpleNamespace(args=SimpleNamespace(caption_v=['a', 'b']), lost_obs=0)
        G = np.zeros((1, 2, 2))
        for H in (np.zeros((2, 2)), -np.ones((2, 2))):
            if not np.any(H):
                with self.assertWarnsRegex(RuntimeWarning, 'No curvature information'):
                    result = sandwich(H, G, np.zeros(2), None, panel, 0)
            else:
                result = sandwich(H, G, np.zeros(2), None, panel, 0)
            self.assertEqual(len(result), 4)
            for array, shape in zip(result, ((2,), (2,), (2, 2), (2, 2))):
                self.assertEqual(array.shape, shape)
                self.assertTrue(np.isnan(array).all())

    def test_two_way_random_effects_use_opposite_dimension_counts(self):
        rng = np.random.default_rng(92)
        rows = []
        for entity, count in enumerate((6, 5, 4, 3)):
            for time in range(count):
                x = rng.normal()
                y = 0.4 + 0.7 * x + rng.normal()
                rows.append((entity, time, x, y))
        data = pd.DataFrame(rows, columns=('id', 'time', 'x', 'y'))
        result = pt.Model('y ~ x', data, entity='id', time='time').fit(
            order=(0, 0, 0), garch_order=(0, 0),
            effects=pt.Effects(group='random', time='random'),
            suppress_output=True,
        )
        panel = result.panel
        group = REObj(panel, True, panel.T_i, panel.T_i, 2)
        time = REObj(panel, False, panel.date_count_mtrx, panel.date_count, 2)
        group_inverse_count = 1 / np.mean(panel.T_i)
        time_inverse_count = 1 / np.mean(panel.date_count)

        self.assertAlmostEqual(group.avg_Tinv, group_inverse_count)
        self.assertAlmostEqual(group.avg_Ninv, time_inverse_count)
        self.assertAlmostEqual(time.avg_Tinv, time_inverse_count)
        self.assertAlmostEqual(time.avg_Ninv, group_inverse_count)

        group.RE(panel.Y, panel)
        time.RE(panel.Y, panel)
        slope_count = panel.X.shape[2] - int(panel.input.has_intercept)
        expected_dof = (
            1 - group_inverse_count - time_inverse_count
            + (1 - slope_count) / panel.NT
        )
        self.assertAlmostEqual(group._dof, expected_dof)
        self.assertAlmostEqual(time._dof, expected_dof)

    def test_two_way_random_effects_reject_zero_variance_theta(self):
        rng = np.random.default_rng(93)
        rows = []
        for entity, count in enumerate((6, 5, 4, 3)):
            for time in range(count):
                x = rng.normal()
                rows.append((entity, time, x, 0.4 + 0.7 * x + rng.normal()))
        data = pd.DataFrame(rows, columns=('id', 'time', 'x', 'y'))
        result = pt.Model('y ~ x', data, entity='id', time='time').fit(
            order=(0, 0, 0), garch_order=(0, 0),
            effects=pt.Effects(group='random', time='random'),
            suppress_output=True,
        )
        panel = result.panel
        group = REObj(panel, True, panel.T_i, panel.T_i, 2)

        with self.assertRaisesRegex(FloatingPointError, 'undefined theta'):
            group.RE(np.zeros_like(panel.Y), panel)

    def test_two_way_random_effect_derivatives_match_finite_differences(self):
        rng = np.random.default_rng(94)
        rows = []
        for entity, count in enumerate((8, 7, 6, 5, 4)):
            alpha = rng.normal()
            for time in range(count):
                x = rng.normal()
                y = 0.4 + 0.7 * x + alpha + 0.2 * time + rng.normal()
                rows.append((entity, time, x, y))
        data = pd.DataFrame(rows, columns=('id', 'time', 'x', 'y'))
        result = pt.Model('y ~ x', data, entity='id', time='time').fit(
            order=(0, 0, 0), garch_order=(0, 0),
            effects=pt.Effects(group='random', time='random'),
            constraints={},  # Exercise likelihood derivatives, not the static estimator.
            suppress_output=True,
        )
        panel = result.panel
        values = np.asarray(panel.Y)
        included = panel.included[3]
        directions = rng.normal(size=values.shape[:2] + (2,)) * included
        step = 1e-5

        for is_group in (True, False):
            def make_effect():
                if is_group:
                    return REObj(panel, True, panel.T_i, panel.T_i, 2)
                return REObj(panel, False, panel.date_count_mtrx, panel.date_count, 2)

            effect = make_effect()
            effect.RE(values, panel)
            first = effect.dRE(directions, panel)
            second = effect.ddRE(directions, panel)

            for index in range(directions.shape[2]):
                direction = directions[:, :, index:index + 1]
                plus = make_effect()
                plus.RE(values + step * direction, panel)
                transformed_plus = plus.RE(values + step * direction, panel, recalc=False)
                minus = make_effect()
                minus.RE(values - step * direction, panel)
                transformed_minus = minus.RE(values - step * direction, panel, recalc=False)
                numeric_first = (transformed_plus - transformed_minus) / (2 * step)
                np.testing.assert_allclose(
                    first[:, :, index:index + 1], numeric_first, rtol=2e-5, atol=2e-7
                )

                plus = make_effect()
                plus.RE(values + step * direction, panel)
                first_plus = plus.dRE(directions, panel)
                minus = make_effect()
                minus.RE(values - step * direction, panel)
                first_minus = minus.dRE(directions, panel)
                numeric_second = (first_plus - first_minus) / (2 * step)
                np.testing.assert_allclose(
                    second[:, :, :, index], numeric_second, rtol=2e-4, atol=2e-6
                )

        args = np.asarray(result._summary.ll.args.args_v, dtype=float)
        args += np.linspace(0.01, 0.04, len(args))

        def score_at(candidate_args):
            likelihood = LL(candidate_args, panel)
            gradient_obj = gradient(panel)
            score, _ = gradient_obj.get(likelihood)
            return score, gradient_obj, likelihood

        analytic_gradient, gradient_obj, likelihood = score_at(args)
        analytic_hessian = hessian(panel, gradient_obj).get(
            likelihood, *likelihood.llfunc.hessian()
        )
        step = 1e-6
        numerical_gradient = np.empty_like(analytic_gradient)
        numerical_hessian = np.empty_like(analytic_hessian)
        for index in range(len(args)):
            plus_args = args.copy()
            minus_args = args.copy()
            plus_args[index] += step
            minus_args[index] -= step
            plus_score, _, _ = score_at(plus_args)
            minus_score, _, _ = score_at(minus_args)
            numerical_gradient[index] = (
                LL(plus_args, panel).LL - LL(minus_args, panel).LL
            ) / (2 * step)
            numerical_hessian[index] = (plus_score - minus_score) / (2 * step)

        np.testing.assert_allclose(
            analytic_gradient, numerical_gradient, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            analytic_hessian, numerical_hessian, rtol=2e-4, atol=2e-5
        )

    def test_native_library_name_matches_platform(self):
        self.assertEqual(_library_name('win32'), 'ctypes.dll')
        self.assertEqual(_library_name('darwin'), 'ctypes.dylib')
        self.assertEqual(_library_name('linux'), 'ctypes.so')
        with self.assertRaisesRegex(ImportError, 'Unsupported platform'):
            _library_name('freebsd')

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_symbolic_h_function_resolves_symbols_by_name(self):
        expression = sp.log(sp.Symbol('e', positive=True)**2 + 1e-8)

        functions = build_h_function(expression)

        self.assertAlmostEqual(float(functions['h_e_val'](1.0, 0.0)), 2.0, places=6)
        with self.assertRaisesRegex(ValueError, 'unsupported symbols: w'):
            build_h_function(sp.Symbol('e') + sp.Symbol('w'))

    def test_model_rejects_removed_multiprocess_interface(self):
        data = pd.DataFrame({'id': [1, 1, 2, 2], 't': [1, 2, 1, 2], 'y': [1.0, 2.0, 3.0, 4.0]})

        self.assertIsInstance(api.Model('y', data, entity='id', time='t'), api.Model)
        with self.assertRaisesRegex(TypeError, 'multiprocess'):
            api.Model('y', data, entity='id', time='t', multiprocess=True)

    def test_tobit_limits_allow_finite_real_cutoffs(self):
        zero_cutoff = FitOptions(tobit_limits=(0, None)).to_engine_options()
        negative_cutoff = FitOptions(tobit_limits=(-2.0, 3.0)).to_engine_options()

        self.assertEqual(zero_cutoff.tobit_limits, [0, None])
        self.assertEqual(negative_cutoff.tobit_limits, [-2.0, 3.0])
        with self.assertRaisesRegex(ValueError, 'lower.*upper'):
            FitOptions(tobit_limits=(3.0, -2.0)).to_engine_options()

    def test_left_censored_fit_at_zero_computes_hessian(self):
        rng = np.random.default_rng(71)
        entity = np.repeat(np.arange(5), 24)
        time = np.tile(np.arange(24), 5)
        x = rng.normal(size=len(entity))
        y = np.maximum(0.3 + 0.5 * x + rng.normal(size=len(entity)), 0)
        data = pd.DataFrame({'id': entity, 't': time, 'x': x, 'y': y})

        result = pt.Model('y ~ x', data, entity='id', time='t').fit(
            order=(0, 0, 0), garch_order=(0, 0), tobit_limits=(0, None),
            suppress_output=True,
        )

        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.params.to_numpy()).all())

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_fit_options_forwards_symbolic_h_function(self):
        e = sp.Symbol('e')
        expression = sp.log(e**2 + 1e-8)

        engine_options = FitOptions(h_function=expression).to_engine_options()

        self.assertIs(engine_options.h_function, expression)

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_fit_applies_symbolic_h_function_to_python_and_cpp_paths(self):
        rng = np.random.default_rng(8)
        entity = np.repeat(np.arange(4), 20)
        time = np.tile(np.arange(20), 4)
        x = rng.normal(size=len(entity))
        y = 0.3 + 0.6 * x + rng.normal(size=len(entity))
        data = pd.DataFrame({'id': entity, 't': time, 'x': x, 'y': y})
        e, _ = sp.symbols('e z')
        expression = sp.log(e**2 + 1e-8)

        result = pt.Model('y ~ x', data, entity='id', time='t').fit(
            order=(0, 0, 0), garch_order=(1, 1), h_function=expression,
            suppress_output=True,
        )

        self.assertIn(b'log', result.panel.h_func_cpp)
        actual = result.panel.h_func(np.array([1.0]), np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(actual, np.log(1.0 + 1e-8))

    def test_get_names_accepts_lists_and_tuples(self):
        data = pd.DataFrame({'x': [1], 'z': [2]})

        self.assertEqual(get_names(['x', 'z'], data, 'variables'), ['x', 'z'])
        self.assertEqual(get_names(('z', 'x'), data, 'variables'), ['x', 'z'])

    def test_pool_func_groups_and_aggregates_rows(self):
        data = pd.DataFrame({'id': [1, 1, 2], 'x': [2.0, 4.0, 8.0]})

        actual = pool_func(data, ('id', 'mean'))
        expected = pd.DataFrame({'x': [3.0, 8.0]}, index=pd.Index([1, 2], name='id'))

        pd.testing.assert_frame_equal(actual, expected)

    def test_forecast_returns_only_future_dependent_values(self):
        rng = np.random.default_rng(22)
        entity = np.repeat(np.arange(4), 20)
        time = np.tile(np.arange(20), 4)
        x = rng.normal(size=len(entity))
        y = 0.3 + 0.5 * x + rng.normal(size=len(entity))
        data = pd.DataFrame({'id': entity, 't': time, 'x': x, 'y': y})

        result = pt.Model('y ~ x', data, entity='id', time='t').fit(
            order=(0, 0, 0), garch_order=(0, 0), constraints={}, suppress_output=True
        )
        predictions = result.predict()
        expected = predictions.loc[predictions['Predicted y'].notna(), 'Predicted y'].to_numpy()
        actual = result.forecast(1)

        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(predictions.loc[predictions['Predicted y'].notna()].index.get_level_values('t').nunique(), 1)
        with self.assertRaisesRegex(ValueError, 'Only 1 future step is available'):
            result.forecast(2)

        ci_95 = result.conf_int(alpha=0.05)
        ci_99 = result.conf_int(alpha=0.01)
        finite = np.isfinite(ci_95.to_numpy()).all(axis=1) & np.isfinite(ci_99.to_numpy()).all(axis=1)
        self.assertTrue(np.any(finite))
        self.assertTrue(np.any(
            np.abs(ci_99.to_numpy()[finite, 1] - ci_99.to_numpy()[finite, 0])
            > np.abs(ci_95.to_numpy()[finite, 1] - ci_95.to_numpy()[finite, 0])
        ))


if __name__ == '__main__':
    unittest.main()