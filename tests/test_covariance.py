import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import lfilter

import paneltime as pt
from paneltime.likelihood.calculus import gradient, hessian
from paneltime.likelihood.covariance import GaussianCovariance, LowRankGaussianCovariance
from paneltime.likelihood.main import LL, pred_u
from paneltime.main import Input
from paneltime.options import FitOptions
from paneltime.processing.panel import Panel


class CovarianceTests(unittest.TestCase):
    @staticmethod
    def data(n=3, t=22):
        rng = np.random.default_rng(42)
        return pd.DataFrame({'id': np.repeat(np.arange(n), t),
                             't': np.tile(np.arange(t), n),
                             'y': rng.normal(size=n*t), 'x': rng.normal(size=n*t)})

    def panel(self, order=(1, 1, 0), garch=(0, 0), effects=None, data=None, exact=True):
        options = FitOptions(order=order, garch_order=garch, exact_ll=exact,
                             effects=effects or pt.Effects()).to_engine_options()
        panel = Panel(Input(self.data() if data is None else data,
                            'y ~ x', 'id', 't', options, None, None), options)
        panel.init()
        return panel

    @staticmethod
    def args(panel):
        args = panel.args.args_init.args_v.copy()
        for name, value in (('rho', .2), ('lambda', -.1), ('gamma', .3),
                            ('psi', .1), ('omega', .8)):
            args[list(panel.args.positions[name])] = value
        return args

    @patch('paneltime.likelihood.main.covariance_likelihood', GaussianCovariance)
    def test_covariance_matches_independent_impulses_and_group_dimensions(self):
        for order in ((0, 0, 0), (1, 1, 0), (2, 2, 0), (0, 2, 1)):
            data = self.data()
            data = data[~((data.id == 0) & (data.t >= 18))]
            panel = self.panel(order=order, data=data)
            ll = LL(self.args(panel), panel)
            self.assertIsNotNone(ll.LL, ll.err_msg)
            reference = 0.
            for group, times, state, e, v, L, K, alpha, weight in ll.covariance.blocks:
                impulse = np.zeros(len(times))
                impulse[0] = 1
                psi = lfilter(np.r_[1., state.ma], np.r_[1., -state.ar], impulse)
                np.testing.assert_allclose(state.psi, psi, atol=1e-12)
                gamma = np.array([psi[:len(psi)-k] @ psi[k:] for k in range(len(psi))])
                S = .8*toeplitz(gamma)
                u = ll.u[group, times, 0]
                reference -= .5*(len(u)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]
                                  + u@np.linalg.solve(S, u))
                np.testing.assert_allclose(np.diag(L@L.T), np.diag(np.diag(v)+v[0]*state.W))
            self.assertAlmostEqual(ll.LL, reference, places=10)
            self.assertEqual(ll.covariance.group_ll.shape, (panel.N,))
            self.assertEqual(ll.covariance.covariance_lags.shape, panel.Y.shape[:2])
            self.assertAlmostEqual(ll.LL, ll.covariance.group_ll.sum(), places=10)
            conditional = LL(self.args(panel), self.panel(order=order, data=data, exact=False))
            self.assertAlmostEqual(ll.LL, conditional.LL+ll.covariance.group_correction.sum(), places=10)

    def test_analytic_scores_and_hessians_with_effects_garch_and_differences(self):
        for order, garch, mode in (
            ((0, 0, 0), (1, 1), 'none'),
            ((2, 2, 0), (2, 2), 'none'),
            ((1, 2, 0), (1, 1), 'fixed'),
            ((2, 1, 1), (1, 1), 'random'),
            ((1, 1, 1), (0, 0), 'none'),
            ((1, 1, 2), (0, 0), 'none'),
        ):
            with self.subTest(order=order, mode=mode):
                data = self.data()
                data = data[~((data.id == 0) & (data.t >= 18))]
                panel = self.panel(order, garch, pt.Effects(group=mode, time=mode), data)
                args = self.args(panel)
                ll = LL(args, panel)
                self.assertIsNotNone(ll.LL, ll.err_msg)
                g = gradient(panel)
                score, scores = g.get(ll)
                H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
                for i in range(len(args)):
                    step = 2e-6
                    up, down = args.copy(), args.copy()
                    up[i] += step
                    down[i] -= step
                    a, b = LL(up, panel), LL(down, panel)
                    self.assertIsNotNone(a.LL, a.err_msg)
                    self.assertIsNotNone(b.LL, b.err_msg)
                    self.assertAlmostEqual(score[i], (a.LL-b.LL)/(2*step), delta=2e-5)
                    np.testing.assert_allclose(scores[:, :, i],
                        (a.ll_value-b.ll_value)[:, :, 0]/(2*step), atol=2e-5, rtol=2e-5)
                    numeric = (gradient(panel).get(a)[0]-gradient(panel).get(b)[0])/(2*step)
                    np.testing.assert_allclose(H[:, i], numeric, atol=3e-4, rtol=2e-5)

    def test_default_and_recursions_are_unchanged(self):
        self.assertIsNone(FitOptions().exact_ll)
        self.assertFalse(FitOptions().to_engine_options().exact_ll)
        effects = pt.Effects(group='random', time='fixed')
        a = self.panel(garch=(1, 1), effects=effects, exact=False)
        b = self.panel(garch=(1, 1), effects=effects)
        x, y = LL(self.args(a), a), LL(self.args(b), b)
        for name in ('e', 'e_RE', 'h', 'var', 'effect_i', 'effect_t'):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
        self.assertIsNone(x.covariance)

    def test_automatic_selection_and_explicit_overrides(self):
        for garch in ((0, 0), (1, 1), (0, 1), (1, 0)):
            for mode in ('none', 'fixed', 'random'):
                for choice in (None, False, True):
                    with self.subTest(garch=garch, mode=mode, choice=choice):
                        panel = self.panel(
                            garch=garch, effects=pt.Effects(group=mode, time=mode),
                            exact=choice)
                        expected = not any(garch) if choice is None else choice
                        self.assertEqual(panel.options.exact_ll, expected)
                        ll = LL(self.args(panel), panel)
                        self.assertIsNotNone(ll.LL, ll.err_msg)
                        self.assertEqual(ll.covariance is not None, expected)

    def test_automatic_selection_preserves_unsupported_conditional_models(self):
        for keyword in ({'vol': 'EGARCH'}, {'fractional_d': .2},
                        {'tobit_limits': (0., None)},
                        {'optimizer': pt.OptimizerOptions(kurtosis_adj=1)}):
            with self.subTest(keyword=keyword):
                options = FitOptions(garch_order=(0, 0), **keyword)
                self.assertFalse(options.to_engine_options().exact_ll)

    def test_public_fit_and_evaluator_resolve_automatic_choice(self):
        for individuals in (1, 3):
            model = pt.Model('y ~ x', self.data(n=individuals), entity='id', time='t')
            for choice in (None, False, True):
                options = FitOptions(order=(0, 1, 0), garch_order=(0, 0),
                                     exact_ll=choice)
                evaluator = model.prepare_likelihood(options)
                expected = choice is not False
                self.assertEqual(evaluator._panel.options.exact_ll, expected)
                result = model.fit(order=options.order, garch_order=(0, 0),
                                   exact_ll=choice)
                self.assertEqual(result.exact_ll, expected)

    def test_static_models_remain_conditional_unless_forced(self):
        for choice in (None, False, True):
            options = FitOptions(order=(0, 0, 0), garch_order=(0, 0), exact_ll=choice)
            self.assertEqual(options.to_engine_options().exact_ll, choice is True)

    @patch('paneltime.likelihood.main.covariance_likelihood', GaussianCovariance)
    def test_seasonal_ma13_score_hessian_and_covariance(self):
        panel = self.panel(order=(0, 13, 0), data=self.data(n=1, t=40))
        args = self.args(panel)
        positions = list(panel.args.positions['lambda'])
        args[positions] = 0
        args[positions[0]], args[positions[11]], args[positions[12]] = -.4, -.55, .22
        ll = LL(args, panel)
        self.assertIsNotNone(ll.LL, ll.err_msg)
        g = gradient(panel)
        score, _ = g.get(ll)
        H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
        state = ll.covariance.blocks[0][2]
        psi = np.zeros(40)
        psi[[0, 1, 12, 13]] = [1., -.4, -.55, .22]
        np.testing.assert_allclose(state.psi, psi, atol=1e-13)
        for i in [0, positions[0], positions[5], positions[11], positions[12], len(args)-1]:
            step = 2e-6
            up, down = args.copy(), args.copy()
            up[i] += step
            down[i] -= step
            a, b = LL(up, panel), LL(down, panel)
            self.assertAlmostEqual(score[i], (a.LL-b.LL)/(2*step), delta=2e-5)
            numeric = (gradient(panel).get(a)[0]-gradient(panel).get(b)[0])/(2*step)
            np.testing.assert_allclose(H[:, i], numeric, atol=3e-4, rtol=2e-5)

    def test_forecast_whitening_and_fit_inference(self):
        panel = self.panel(order=(0, 1, 0), data=self.data(n=1, t=12))
        ll = LL(self.args(panel), panel)
        ll.predict(panel)
        ma = -.1
        gamma = np.r_[.8*(1+ma*ma), .8*ma, np.zeros(11)]
        S = toeplitz(gamma)
        cross = S[-1, :-1]
        u = ll.u[0, :, 0]
        mean = cross@np.linalg.solve(S[:-1, :-1], u)
        var = S[-1, -1]-cross@np.linalg.solve(S[:-1, :-1], cross)
        conditional = pred_u(ll.u, ll.e, ll.args.args_d['rho'], ll.args.args_d['lambda'], panel)
        correction, _ = ll.covariance.forecast_adjustment()
        np.testing.assert_allclose(conditional+correction, mean, atol=1e-12)
        np.testing.assert_allclose(ll.var_pred, var, atol=1e-12)
        np.testing.assert_allclose(ll.Y_st-ll.X_st@ll.args.args_d['beta'], ll.e_norm, atol=1e-12)
        result = pt.Model('y ~ x', self.data(), entity='id', time='t').fit(
            order=(0, 1, 0), garch_order=(0, 0), exact_ll=True,
            optimizer=pt.OptimizerOptions(max_iterations=300, garch_ridge_wald=0))
        self.assertTrue(result.converged)
        self.assertTrue(result.exact_ll)
        self.assertTrue(np.isfinite(result.bse).all())
        self.assertTrue(np.isfinite(result.robust_bse).all())
        self.assertTrue(pickle.loads(pickle.dumps(result)).exact_ll)

    def test_restrictions(self):
        for keyword in ({'vol': 'EGARCH'}, {'fractional_d': .2},
                        {'tobit_limits': (0., None)}, {'exact_ll': 'yes'}):
            with self.assertRaises((ValueError, TypeError)):
                FitOptions(**{'exact_ll': True, **keyword}).to_engine_options()
        panel = self.panel()
        args = self.args(panel)
        args[list(panel.args.positions['rho'])] = 1.01
        ll = LL(args, panel)
        self.assertIsNone(ll.LL)
        self.assertIn('stationary AR roots', ll.err_msg)

    @patch('paneltime.likelihood.main.covariance_likelihood', GaussianCovariance)
    def test_truncated_arma_forecast_matches_finite_impulse_covariance(self):
        panel = self.panel(data=self.data(n=1, t=15))
        args = self.args(panel)
        args[list(panel.args.positions['rho'])] = .8
        ll = LL(args, panel)
        self.assertIsNotNone(ll.LL, ll.err_msg)
        ll.predict(panel)
        psi = ll.covariance.blocks[0][2].psi
        gamma = np.r_[ll.covariance.blocks[0][2].autocovariances, 0.]*.8
        S = toeplitz(gamma)
        u = ll.u[0, :, 0]
        cross = S[-1, :-1]
        expected = cross@np.linalg.solve(S[:-1, :-1], u)
        expected_var = S[-1, -1]-cross@np.linalg.solve(S[:-1, :-1], cross)
        conditional = pred_u(ll.u, ll.e, ll.args.args_d['rho'], ll.args.args_d['lambda'], panel)
        correction, _ = ll.covariance.forecast_adjustment()
        np.testing.assert_allclose(conditional+correction, expected, atol=1e-10)
        np.testing.assert_allclose(ll.var_pred, expected_var, atol=1e-10)
