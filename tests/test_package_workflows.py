import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

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
from paneltime.maximization.dfpmax import LR_LIMIT
from paneltime.maximization.constraints import arma_resultant
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

    def test_first_garch_variance_is_fixed_sample_variance(self):
        panel = self._presample_panel((1, 0, 1), (1, 1))
        positions = panel.args.positions
        args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        args[list(positions['omega'])] = 0.6
        args[list(positions['gamma'])] = 0.3
        args[list(positions['psi'])] = 0.1
        ll = LL(args, panel)
        np.testing.assert_allclose(ll.var[:, 0, 0], panel.args.init_var)
        self.assertNotIn('initvar', panel.args.names_v)

        args[list(positions['omega'])] = 0.2
        np.testing.assert_allclose(LL(args, panel).var[:, 0, 0], panel.args.init_var)
        args[list(positions['gamma'])] = 0.95
        args[list(positions['psi'])] = 0.2
        ll = LL(args, panel)
        self.assertIsNotNone(ll.LL)
        np.testing.assert_allclose(ll.var[:, 0, 0], panel.args.init_var)

    def test_zero_garch_coefficients_keep_only_the_first_variance_fixed(self):
        for order, garch in (((0, 0, 0), (0, 0)), ((1, 0, 1), (0, 0)), ((0, 0, 0), (1, 1))):
            panel = self._presample_panel(order, garch, seed=5, n=4, t=10)
            args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
            for name in ('gamma', 'psi'):
                args[list(panel.args.positions.get(name, []))] = 0.0
            args[list(panel.args.positions['omega'])] = 0.7
            ll = LL(args, panel)
            expected = np.full_like(ll.var, 0.7)
            if any(garch):
                expected[:, 0, 0] = panel.args.init_var
            np.testing.assert_allclose(ll.var, expected)

    def test_garch_recursion_uses_fixed_first_variance_and_zero_missing_lags(self):
        panel = self._presample_panel((0, 0, 0), (2, 2), n=4, t=10)
        positions = panel.args.positions
        args = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        gamma, psi = [0.3, 0.1], [0.1, 0.05]
        args[list(positions['gamma'])] = gamma
        args[list(positions['psi'])] = psi
        args[list(positions['omega'])] = 0.5
        ll = LL(args, panel)
        expected = np.full_like(ll.var, 0.5)
        expected[:, 0, 0] = panel.args.init_var
        for t in range(1, ll.var.shape[1]):
            for j in range(1, 3):
                prior_var = expected[:, t-j, 0] if t >= j else 0
                prior_h = ll.h_val[:, t-j, 0] if t >= j else 0
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
        np.testing.assert_allclose(g.dvar_omega[:, 0], 0)
        np.testing.assert_allclose(g.dvar_gamma[:, 0], 0)
        np.testing.assert_allclose(g.dvar_psi[:, 0], 0)
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

    def test_white_noise_arma_common_factor_is_removed(self):
        for seed, garch in [(1, (0, 0)), (2, (0, 0)), (2, (1, 1))]:
            with self.subTest(seed=seed, garch=garch):
                data = self._arma_panel(0.0, 0.0, seed)
                model = pt.Model('y ~ x', data, entity='id', time='time')
                result = model.fit(order=(1, 1, 0), garch_order=garch, suppress_output=True)
                restricted = model.fit(
                    order=(1, 1, 0), garch_order=garch, constraints={'rho': 0.0, 'lambda': 0.0},
                    suppress_output=True,
                )
                self.assertTrue(np.all(np.isfinite(result.pvalues.iloc[:2])))
                if 'common factor' in result.optim_result.msg:
                    np.testing.assert_array_equal(result.params.iloc[2:4], 0.0)
                    self.assertAlmostEqual(result.llf, restricted.llf, delta=1e-3)
                else:
                    from paneltime.maximization.constraints import Constraints
                    with patch.object(Constraints, '_ci_supports', return_value=True):
                        wald_only = model.fit(
                            order=(1, 1, 0), garch_order=garch, suppress_output=True,
                        )
                    self.assertTrue(
                        'common factor' in wald_only.optim_result.msg
                        or 2*(wald_only.llf - restricted.llf) > LR_LIMIT,
                    )
        data = self._arma_panel(0.0, 0.0, 1)
        result = pt.Model('y ~ x', data, entity='id', time='time').fit(
            order=(1, 1, 0), suppress_output=True)
        self.assertNotIn('common factor', result.optim_result.msg)

    def test_combined_detectors_require_matching_ci_and_wald(self):
        from paneltime.maximization.constraints import Constraints

        panel = self._presample_panel((1, 1, 0), (2, 2), seed=11, n=6, t=14)
        x = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        rho = panel.args.positions['rho'][0]
        lam = panel.args.positions['lambda'][0]
        g0, g1 = panel.args.positions['gamma']
        beta0, beta1 = panel.args.positions['beta'][:2]
        x[rho], x[lam], x[g0], x[g1] = 0.5, -0.5, 0.5, 0.0
        constraints = Constraints(panel, x, 0, panel.options.ARMA_constraint)

        def correlated(H, indices, scale=1):
            H[np.ix_(indices, indices)] = -scale*np.array(
                [[1, 1 - 1e-8], [1 - 1e-8, 1]],
            )

        H = -np.eye(len(x))
        correlated(H, [beta0, beta1])
        self.assertIsNone(constraints.arma_cancellation(x, H))
        self.assertIsNone(constraints.gamma_ridge(x, H))
        H = -np.eye(len(x))
        H[g1, g1] = -1e-8
        self.assertIsNone(constraints.gamma_ridge(x, H))
        self.assertIsNone(constraints.arma_cancellation(x, H))
        correlated(H, [rho, lam])
        correlated(H, [g0, g1])
        self.assertEqual(constraints.arma_cancellation(x, H), [rho, lam])
        self.assertEqual(constraints.gamma_ridge(x, H), [g1])
        correlated(H, [g0, g1], scale=1e12)
        self.assertIsNone(constraints.gamma_ridge(x, H))
        x[lam] = 10.0
        self.assertIsNone(constraints.arma_cancellation(x, H))
        constraints.multicoll_threshold_report = 1e10
        self.assertIsNone(constraints.arma_cancellation(x, H))
        self.assertIsNone(constraints.gamma_ridge(x, H))
        constraints.multicoll_threshold_report = 30
        constraints.add(g1, None, 'user constraints', value=0.0)
        self.assertIsNone(constraints.gamma_ridge(x, H))

    def test_arma_resultant_detects_shared_roots(self):
        # (1 - .5L)(1 - .3L) and (1 - .5L)(1 + .2L) share the factor (1 - .5L).
        self.assertAlmostEqual(arma_resultant([.8, -.15], [-.3, -.1]), 0.0, places=12)
        self.assertAlmostEqual(arma_resultant([.5], [.3]), .8, places=12)

    def test_gamma_ridge_detector_flags_weak_split(self):
        panel = self._presample_panel((0, 0, 0), (2, 2), seed=11, n=6, t=14)
        x = np.asarray(panel.args.args_init.args_v, dtype=float)
        g0, g1 = panel.args.positions['gamma']
        x[g0], x[g1] = 0.92, 0.00
        H = -np.eye(len(x))
        H[np.ix_([g0, g1], [g0, g1])] = [[-1, -1 + 1e-8], [-1 + 1e-8, -1]]
        ll = LL(x, panel)
        from paneltime.maximization.computation import Computation
        c = Computation(x, panel, panel.options.tolerance, 4*np.finfo(float).eps, True)
        c.calc_gradient(ll)
        c.set(0, 0.0, 1.0, False, H, ll, x, panel.options.ARMA_constraint)
        self.assertEqual(c.constr.gamma_ridge(x, H), [g1])

        H[g1, g1] = -100.0
        self.assertIsNone(c.constr.gamma_ridge(x, H))

    def test_garch_ridge_wald_option(self):
        from paneltime.options import create_engine_options

        self.assertEqual(pt.OptimizerOptions().garch_ridge_wald, 0.2)
        self.assertEqual(create_engine_options().GARCH_ridge_wald, 0.2)
        for value in (0, 0.2, 1.0):
            with self.subTest(value=value):
                engine = pt.FitOptions(
                    optimizer=pt.OptimizerOptions(garch_ridge_wald=value),
                ).to_engine_options()
                self.assertEqual(engine.GARCH_ridge_wald, value)
        for value in (-1, float('nan'), float('inf'), -float('inf'), True, None, '0.2'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'optimizer.garch_ridge_wald'):
                    pt.OptimizerOptions(garch_ridge_wald=value).validate()
                engine = create_engine_options()
                engine.GARCH_ridge_wald = value
                with self.assertRaisesRegex(ValueError, 'optimizer.garch_ridge_wald'):
                    engine.validate()

    def test_gamma_ridge_uses_configured_squared_wald_threshold(self):
        from paneltime.maximization.constraints import Constraints

        panel = self._presample_panel((0, 0, 0), (2, 2), seed=11, n=6, t=14)
        x = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        g0, g1 = panel.args.positions['gamma']
        x[g0], x[g1] = 0.5, 0.0
        H = -np.eye(len(x))
        # W = 0.5**2 / 2 = 0.125, between the old and current defaults.
        correlation = np.sqrt(1 - 1 / 2000)
        H[np.ix_([g0, g1], [g0, g1])] = -1000 * np.array(
            [[1, correlation], [correlation, 1]],
        )
        wald = x[g0]**2 / np.linalg.pinv(-H)[g1, g1]
        for threshold, expected in ((0, None), (0.1, None), (wald, None),
                                    (0.2, [g1]), (1.0, [g1])):
            with self.subTest(threshold=threshold):
                panel.options.GARCH_ridge_wald = threshold
                constraints = Constraints(panel, x, 0, panel.options.ARMA_constraint)
                self.assertEqual(constraints.gamma_ridge(x, H), expected)
        self.assertIsNone(constraints.gamma_ridge(x, H, 0))
        for scale, expected in ((0.0, None), (-1.0, [g1]), (-100.0, None)):
            with self.subTest(scale=scale):
                trial = H.copy()
                trial[np.ix_([g0, g1], [g0, g1])] *= scale
                self.assertEqual(constraints.gamma_ridge(x, trial), expected)
                self.assertIsNone(constraints.gamma_ridge(x, trial, 0))

    def test_option_documentation_matches_schema(self):
        from setup_script import write_option_docs

        with TemporaryDirectory() as directory:
            generated = Path(directory) / 'options.qmd'
            write_option_docs(generated)
            checked_in = Path(__file__).resolve().parents[1] / 'qmd' / 'options.qmd'
            self.assertEqual(
                generated.read_text(encoding='utf-8'),
                checked_in.read_text(encoding='utf-8'),
            )

    def test_fit_routes_garch_ridge_threshold_and_zero_disables_it(self):
        from paneltime.maximization.constraints import Constraints

        model = pt.Model('y ~ x', self._arma_panel(0.5, -0.3, 3), entity='id', time='time')
        for threshold in (0, 0.2, 1.0):
            with self.subTest(threshold=threshold):
                with patch.object(Constraints, 'gamma_ridge', return_value=None) as detector:
                    result = model.fit(
                        order=(1, 1, 0), garch_order=(2, 2),
                        optimizer=pt.OptimizerOptions(garch_ridge_wald=threshold),
                        suppress_output=True,
                    )
                self.assertTrue(result.converged)
                if threshold == 0:
                    detector.assert_not_called()
                else:
                    detector.assert_called_once()
                    self.assertEqual(detector.call_args.args[2], threshold)

    def test_gamma_release_accepts_only_identified_likelihood_preserving_retry(self):
        from paneltime.maximization.computation import Computation
        from paneltime.maximization.dfpmax import _release_gamma

        panel = self._presample_panel((0, 0, 0), (2, 2), seed=11, n=6, t=14)
        x = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        g0, g1 = panel.args.positions['gamma']
        beta0 = panel.args.positions['beta'][0]
        x[beta0] = 0.0
        panel.args.user_constraints['beta'] = [0.0]
        x[g0], x[g1] = 0.5, 0.0
        comput = Computation(x, panel, panel.options.tolerance, 4*np.finfo(float).eps, True)
        comput.gamma_reduced = [g1]
        comput.set_constr(x, panel.options.ARMA_constraint)
        reduced = dict(x=x, f=-100.0, its=5, dx_norm=0.0, msg='Reduced fit', conv=1)
        original = dict(f=-99.0)
        for case in ('identified', 'ridge', 'indefinite', 'singular', 'stalled',
                     'worse_original', 'worse_reduced', 'nonfinite'):
            with self.subTest(case=case):
                def iteration(retry, actual_panel, state):
                    self.assertIs(actual_panel, panel)
                    self.assertNotIn(g1, retry.constr.fixed)
                    self.assertIn(beta0, retry.constr.fixed)
                    self.assertEqual(retry.gamma_reduced, [])
                    self.assertEqual(state['its'], 5)
                    self.assertEqual(state['step'], 1.0)
                    hessian = -np.eye(len(x))
                    if case == 'ridge':
                        hessian[np.ix_([g0, g1], [g0, g1])] = [
                            [-1, -1 + 1e-8], [-1 + 1e-8, -1],
                        ]
                    elif case == 'indefinite':
                        hessian[g1, g1] = 0.5
                    elif case == 'singular':
                        hessian[g1, g1] = 0
                    value = {'worse_original': -99.5, 'worse_reduced': -101.0,
                             'nonfinite': np.nan}.get(case, -98.0)
                    return dict(
                        state, H=hessian, g=np.zeros(len(x)), f=value, its=8,
                        conv=4 if case == 'stalled' else 1,
                    )

                with patch('paneltime.maximization.dfpmax._iterate', side_effect=iteration) as run:
                    state, final_comput, released = _release_gamma(comput, panel, reduced, original)
                run.assert_called_once()
                self.assertEqual(released, case == 'identified')
                self.assertEqual(state['its'], 8)
                self.assertIn(g1, comput.constr.fixed)
                self.assertEqual(comput.gamma_reduced, [g1])
                if released:
                    self.assertNotIn(g1, final_comput.constr.fixed)
                else:
                    self.assertIs(final_comput, comput)
                    self.assertEqual(state['f'], reduced['f'])
                    self.assertEqual(state['msg'], reduced['msg'])

    def test_gamma_release_respects_budget_and_invalid_initial_likelihood(self):
        from paneltime.maximization.computation import Computation
        from paneltime.maximization.dfpmax import _release_gamma

        panel = self._presample_panel((0, 0, 0), (2, 2), seed=11, n=6, t=14)
        x = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
        g1 = panel.args.positions['gamma'][-1]
        x[g1] = 0
        comput = Computation(x, panel, panel.options.tolerance, 4*np.finfo(float).eps, True)
        comput.gamma_reduced = [g1]
        comput.set_constr(x, panel.options.ARMA_constraint)
        reduced = dict(x=x, f=-100, its=panel.options.max_iterations, dx_norm=0.0)
        with patch('paneltime.maximization.dfpmax._iterate') as run:
            state, final_comput, released = _release_gamma(comput, panel, reduced, reduced)
        run.assert_not_called()
        self.assertIs(state, reduced)
        self.assertIs(final_comput, comput)
        self.assertFalse(released)
        reduced['its'] = 1
        for likelihood in (None, SimpleNamespace(LL=None), SimpleNamespace(LL=np.nan)):
            with self.subTest(likelihood=likelihood):
                with patch('paneltime.maximization.dfpmax.logl.LL', return_value=likelihood), \
                        patch('paneltime.maximization.dfpmax._iterate') as run:
                    state, final_comput, released = _release_gamma(comput, panel, reduced, reduced)
                run.assert_not_called()
                self.assertIs(state, reduced)
                self.assertIs(final_comput, comput)
                self.assertFalse(released)

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

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_symbolic_h_functions_match_native_recursion_and_finite_differences(self):
        from paneltime.likelihood.function import h_values

        e, z = sp.symbols('e z')
        data = self._arma_panel(0.2, 0.1, 3).query('id < 3 and time < 12')
        expressions = (e**2, sp.Integer(1), e**2 + z, sp.exp(z)*e**2,
                       sp.sqrt(e**2 + z**2 + 1))
        for expression in expressions:
            with self.subTest(expression=expression):
                options = FitOptions(
                    order=(1, 1, 0), garch_order=(1, 1), h_function=expression,
                ).to_engine_options()
                panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
                panel.init()
                self.assertEqual(panel.z_active, z in expression.free_symbols)
                x = np.asarray(panel.args.args_init.args_v, dtype=float).copy()
                for group in ('rho', 'lambda', 'gamma', 'psi'):
                    x[list(panel.args.positions[group])] = 0.1
                x[list(panel.args.positions['omega'])] = 1.0
                if panel.z_active:
                    x[list(panel.args.positions['z'])] = 0.2
                ll = LL(x, panel)
                self.assertIsNotNone(ll.LL)
                np.testing.assert_allclose(ll.h, ll.h_val, atol=1e-12)
                shape = ll.args.args_d.get('z')
                np.testing.assert_allclose(h_values(panel, ll.e_RE, shape), ll.h_val)
                g = gradient(panel)
                score, _ = g.get(ll)
                H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
                for i in range(len(x)):
                    up, down = x.copy(), x.copy()
                    up[i] += 1e-5
                    down[i] -= 1e-5
                    lu, ld = LL(up, panel), LL(down, panel)
                    self.assertAlmostEqual(score[i], (lu.LL - ld.LL)/2e-5, delta=1e-4)
                    numeric = (gradient(panel).get(lu)[0] - gradient(panel).get(ld)[0])/2e-5
                    np.testing.assert_allclose(H[:, i], numeric, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_symbolic_h_function_pickle_and_error_messages(self):
        import pickle
        from paneltime.likelihood.symbolic import SymbolicHFunction

        e, z = sp.symbols('e z')
        function = pickle.loads(pickle.dumps(SymbolicHFunction(e**2 + z, 0.2)))
        self.assertTrue(function.has_z)
        np.testing.assert_allclose(function(np.array([1., 2.]), None, None), [1.2, 4.2])
        for expression in (sp.Abs(e), sp.Abs(sp.Symbol('e', real=True))):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, 'unsupported derivative.*smooth expression'):
                    build_h_function(expression)
        data = self._arma_panel(0.0, 0.0, 1)
        with self.assertRaisesRegex(ValueError, 'requires at least one ARCH lag'):
            pt.Model('y ~ x', data, entity='id', time='time').fit(
                order=(0, 0, 0), garch_order=(0, 0), h_function=e**2 + z,
            )

    def test_h_function_native_validation_respects_inputs_and_checks_python(self):
        from paneltime.likelihood.hfunc import get_model_string, test_expr

        self.assertAlmostEqual(test_expr(b'e + z', 2, 4, 3), 5)
        with self.assertRaisesRegex(ValueError, 'Python/C\\+\\+ mismatch'):
            get_model_string(
                SimpleNamespace(h_val_cpp='e^2', z=None, v=1, h=lambda e, e2, v: e**2 + 1),
                2, 4, 3,
            )

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_numeric_custom_likelihood_preserves_symbolic_h_function(self):
        from tests.test_likelihood_models import NumericNormal
        from paneltime.likelihood.symbolic import apply_h_function

        model = NumericNormal(np.array([[[0.5]], [[1.0]]]), np.ones((2, 1, 1)))
        expression = sp.sqrt(sp.Symbol('e')**2 + 1)
        apply_h_function(model, expression)
        expected = model.h_val.copy()
        for method in (model.score, model.hessian):
            method()
            np.testing.assert_allclose(model.h_val, expected)

    @unittest.skipIf(sp is None, 'SymPy is optional')
    def test_shape_h_function_fits_builtin_and_numeric_likelihoods(self):
        from tests.test_likelihood_models import NumericNormal

        e, z = sp.symbols('e z')
        data = self._arma_panel(0.2, 0.1, 3).query('id < 4 and time < 25')
        for vol, likelihood in (('GARCH', None), ('EGARCH', None), ('GARCH', NumericNormal)):
            with self.subTest(vol=vol, likelihood=likelihood):
                result = pt.Model('y ~ x', data, entity='id', time='time').fit(
                    order=(1, 1, 0), garch_order=(1, 1), vol=vol, likelihood=likelihood,
                    h_function=sp.sqrt(e**2 + z**2 + 1),
                    constraints={'z': 0.2, 'gamma': 0.1, 'psi': 0.1},
                    suppress_output=True,
                )
                self.assertTrue(result.converged)
                self.assertTrue(np.isfinite(result.llf))
                self.assertEqual(result.params['z in h(e,z)'], 0.2)

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