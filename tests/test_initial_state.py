import pickle
import importlib.util
import unittest

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import lfilter

import paneltime as pt
from paneltime.likelihood.calculus import gradient, hessian
from paneltime.likelihood.initial_state import StationaryState
from paneltime.likelihood.main import LL, pred_u
from paneltime.main import Input
from paneltime.options import FitOptions
from paneltime.processing.panel import Panel


class InitialStateTests(unittest.TestCase):
    @staticmethod
    def data(n=4, t=26):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'y': rng.normal(size=n*t) + np.repeat(np.linspace(-.8, .8, n), t),
            'x': rng.normal(size=n*t),
        })

    def panel(self, order=(1, 1, 0), garch=(0, 0), effects=None, data=None,
              initialization='stationary', h_function=None):
        data = self.data() if data is None else data
        options = FitOptions(
            order=order, garch_order=garch, effects=effects or pt.Effects(),
            arima_initialization=initialization, h_function=h_function,
        ).to_engine_options()
        panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
        panel.init()
        return panel

    def args(self, panel):
        args = panel.args.args_init.args_v.copy()
        for name, value in (('rho', .12), ('lambda', -.1), ('gamma', .3),
                            ('psi', .1), ('omega', .8), ('z', .2)):
            if name in panel.args.positions:
                args[list(panel.args.positions[name])] = value
        return args

    @staticmethod
    def reference_covariance(ar, ma, length, variance):
        impulse = np.zeros(8192)
        impulse[0] = 1
        response = lfilter(np.r_[1., ma], np.r_[1., -np.asarray(ar)], impulse)
        covariances = np.array([response[:len(response)-lag] @ response[lag:]
                                for lag in range(length)])
        return variance * toeplitz(covariances)

    def test_covariance_likelihood_matches_independent_impulse_reference(self):
        for order in ((0, 1, 0), (1, 0, 0), (2, 2, 0), (0, 0, 0), (1, 1, 1)):
            data = self.data()
            data = data[~((data.id == 0) & (data.time >= 21))]
            panel = self.panel(order=order, data=data)
            ll = LL(self.args(panel), panel)
            reference = 0.
            for group, length in enumerate(panel.T_arr[:, 0]):
                u = ll.u[group, order[2]:length, 0]
                covariance = self.reference_covariance(
                    ll.args.args_d['rho'].ravel(), ll.args.args_d['lambda'].ravel(),
                    len(u), .8,
                )
                reference -= .5 * (len(u)*np.log(2*np.pi)
                                    + np.linalg.slogdet(covariance)[1]
                                    + u @ np.linalg.solve(covariance, u))
            self.assertAlmostEqual(ll.LL, reference, places=10)
            self.assertAlmostEqual(ll.LL, ll.ll_value.sum(), places=12)
            self.assertEqual(panel.NT, sum(panel.T_arr[:, 0] - order[2]))

    def check_derivatives(self, panel):
        args = self.args(panel)

        def evaluate(values):
            ll = LL(values, panel)
            self.assertIsNotNone(ll.LL, ll.err_msg)
            g = gradient(panel)
            score, scores = g.get(ll)
            return ll, g, score, scores

        ll, g, score, scores = evaluate(args)
        H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
        np.testing.assert_allclose(H, H.T, atol=1e-10)
        np.testing.assert_allclose(scores.sum(axis=(0, 1)), score, atol=1e-12)
        step = 2e-6
        for i in range(len(args)):
            up, down = args.copy(), args.copy()
            up[i] += step
            down[i] -= step
            lu, _, gu, _ = evaluate(up)
            ld, _, gd, _ = evaluate(down)
            self.assertAlmostEqual(score[i], (lu.LL-ld.LL)/(2*step), delta=2e-5)
            np.testing.assert_allclose(
                scores[:, :, i], (lu.ll_value-ld.ll_value)[:, :, 0]/(2*step),
                atol=2e-5, rtol=2e-5,
            )
            np.testing.assert_allclose(H[:, i], (gu-gd)/(2*step), atol=3e-4, rtol=2e-5)

    def test_full_score_and_hessian_all_parameter_blocks(self):
        for order, garch in (
            ((1, 1, 0), (0, 0)), ((2, 2, 0), (2, 2)),
            ((0, 1, 0), (1, 1)), ((1, 0, 0), (1, 1)),
            ((0, 0, 0), (1, 1)), ((1, 1, 1), (1, 1)),
        ):
            with self.subTest(order=order, garch=garch):
                self.check_derivatives(self.panel(order=order, garch=garch))

    def test_effects_and_ragged_panel_score_and_hessian(self):
        data = self.data()
        data = data[~((data.id == 0) & (data.time >= 21))]
        for mode in ('fixed', 'random'):
            for garch in ((0, 0), (1, 1)):
                with self.subTest(mode=mode, garch=garch):
                    self.check_derivatives(self.panel(
                        order=(2, 2, 0), garch=garch, data=data,
                        effects=pt.Effects(group=mode, time=mode),
                    ))

    @unittest.skipUnless(importlib.util.find_spec('sympy'), 'SymPy is optional')
    def test_gaussian_variance_shape_cross_derivatives(self):
        import sympy
        e, z = sympy.symbols('e z', real=True)
        self.check_derivatives(self.panel(garch=(1, 1), h_function=e**2 + z))

    def test_seasonal_ma13_likelihood_score_and_hessian(self):
        panel = self.panel(order=(0, 13, 0), data=self.data(n=1, t=40))
        args = self.args(panel)
        positions = list(panel.args.positions['lambda'])
        args[positions] = 0
        args[positions[0]], args[positions[11]], args[positions[12]] = -.4, -.55, .22
        ll = LL(args, panel)
        g = gradient(panel)
        score, _ = g.get(ll)
        H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
        covariance = self.reference_covariance([], args[positions], 40, .8)
        u = ll.u[0, :, 0]
        reference = -.5 * (40*np.log(2*np.pi) + np.linalg.slogdet(covariance)[1]
                            + u @ np.linalg.solve(covariance, u))
        self.assertAlmostEqual(ll.LL, reference, places=10)
        for i in [0, positions[0], positions[5], positions[11], positions[12], len(args)-1]:
            step = 2e-6
            up, down = args.copy(), args.copy()
            up[i] += step
            down[i] -= step
            lu, ld = LL(up, panel), LL(down, panel)
            self.assertAlmostEqual(score[i], (lu.LL-ld.LL)/(2*step), delta=2e-5)
            numeric = (gradient(panel).get(lu)[0]-gradient(panel).get(ld)[0])/(2*step)
            np.testing.assert_allclose(H[:, i], numeric, rtol=2e-5, atol=3e-4)

    def test_stationary_covariance_first_and_second_derivatives(self):
        parameters = np.array([.3, -.12, -.2, .15])
        state = StationaryState(parameters[:2], parameters[2:], 18)
        state.derivatives(2)
        step = 2e-6
        for i in range(4):
            up, down = parameters.copy(), parameters.copy()
            up[i] += step
            down[i] -= step
            a, b = StationaryState(up[:2], up[2:], 18), StationaryState(down[:2], down[2:], 18)
            a.derivatives()
            b.derivatives()
            np.testing.assert_allclose(state.dW[i], (a.W-b.W)/(2*step), atol=1e-9)
            np.testing.assert_allclose(state.ddW[i], (a.dW-b.dW)/(2*step), atol=1e-8)

    def test_derivatives_near_stationary_boundary_and_common_factor(self):
        for ar, ma in ((.97, -.35), (.3, -.3)):
            panel = self.panel(data=self.data(n=1, t=18))
            args = self.args(panel)
            args[list(panel.args.positions['rho'])] = ar
            args[list(panel.args.positions['lambda'])] = ma
            ll = LL(args, panel)
            g = gradient(panel)
            score, _ = g.get(ll)
            H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
            for position in list(panel.args.positions['rho']) + list(panel.args.positions['lambda']):
                step = 1e-6
                up, down = args.copy(), args.copy()
                up[position] += step
                down[position] -= step
                lu, ld = LL(up, panel), LL(down, panel)
                self.assertAlmostEqual(score[position], (lu.LL-ld.LL)/(2*step), delta=2e-5)
                numeric = (gradient(panel).get(lu)[0]-gradient(panel).get(ld)[0])/(2*step)
                np.testing.assert_allclose(H[:, position], numeric, rtol=2e-5, atol=3e-4)

    def test_default_and_explicit_conditional_are_identical(self):
        data = self.data()
        default = FitOptions(order=(1, 1, 0), garch_order=(1, 1))
        explicit = FitOptions(order=(1, 1, 0), garch_order=(1, 1),
                              arima_initialization='conditional')
        panels = []
        for options in (default, explicit):
            engine = options.to_engine_options()
            panel = Panel(Input(data, 'y ~ x', 'id', 'time', engine, None, None), engine)
            panel.init()
            panels.append(panel)
        a, b = [LL(self.args(panel), panel) for panel in panels]
        self.assertIsNone(a.initial_state)
        self.assertEqual(a.LL, b.LL)
        ga, gb = gradient(panels[0]), gradient(panels[1])
        np.testing.assert_array_equal(ga.get(a)[0], gb.get(b)[0])
        np.testing.assert_array_equal(
            hessian(panels[0], ga).get(a, *a.llfunc.hessian()),
            hessian(panels[1], gb).get(b, *b.llfunc.hessian()),
        )

    def test_garch_and_effect_recursions_are_not_changed(self):
        for mode in ('none', 'fixed', 'random'):
            effects = pt.Effects(group=mode, time=mode)
            conditional = self.panel(garch=(1, 1), effects=effects, initialization='conditional')
            stationary = self.panel(garch=(1, 1), effects=effects)
            a, b = LL(self.args(conditional), conditional), LL(self.args(stationary), stationary)
            for name in ('e', 'e_RE', 'effect_i', 'effect_t', 'h', 'var'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
            self.assertNotEqual(a.LL, b.LL)

    def test_forecast_mean_variance_and_whitening_match_gaussian_conditioning(self):
        panel = self.panel(data=self.data(n=1, t=12), order=(1, 1, 0))
        ll = LL(self.args(panel), panel)
        prediction = ll.predict(panel)
        self.assertFalse(prediction.empty)
        u = ll.u[0, :, 0]
        S = self.reference_covariance([.12], [-.1], len(u)+1, .8)
        cross = S[-1, :-1]
        expected = cross @ np.linalg.solve(S[:-1, :-1], u)
        variance = S[-1, -1] - cross @ np.linalg.solve(S[:-1, :-1], cross)
        conditional = pred_u(ll.u, ll.e, ll.args.args_d['rho'], ll.args.args_d['lambda'], panel)
        adjustment, _ = ll.initial_state.forecast_adjustment()
        np.testing.assert_allclose(conditional + adjustment, expected, atol=1e-12)
        np.testing.assert_allclose(ll.var_pred, variance, atol=1e-12)
        np.testing.assert_allclose(np.sum(ll.e_norm**2), u @ np.linalg.solve(S[:-1, :-1], u), atol=1e-12)
        np.testing.assert_allclose(ll.Y_st-ll.X_st @ ll.args.args_d['beta'], ll.e_norm, atol=1e-12)

    def test_invalid_options_and_nonstationary_ar_are_explicit(self):
        for keyword in (
            {'arima_initialization': 'unknown'},
            {'fractional_d': .2}, {'vol': 'EGARCH'}, {'tobit_limits': (0., None)},
            {'optimizer': pt.OptimizerOptions(kurtosis_adj=.1)},
        ):
            options = {'arima_initialization': 'stationary', **keyword}
            with self.subTest(options=options), self.assertRaises(ValueError):
                FitOptions(**options).to_engine_options()
        panel = self.panel()
        args = self.args(panel)
        args[list(panel.args.positions['rho'])] = 1.01
        ll = LL(args, panel)
        self.assertIsNone(ll.LL)
        self.assertIn('stationary AR roots', ll.err_msg)
        data = self.data()
        data.loc[(data.id == 0) & (data.time > 12), 'time'] += 1
        with self.assertRaisesRegex(ValueError, 'regular observations'):
            self.panel(data=data)

    def test_fit_and_serialization_preserve_inference(self):
        result = pt.Model('y ~ x', self.data(), entity='id', time='time').fit(
            order=(1, 1, 0), garch_order=(0, 0), arima_initialization='stationary',
            optimizer=pt.OptimizerOptions(max_iterations=300, garch_ridge_wald=0),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.arima_initialization, 'stationary')
        self.assertIn('stationary ARMA state', str(result))
        self.assertTrue(np.isfinite(result.bse).all())
        self.assertTrue(np.isfinite(result.robust_bse).all())
        restored = pickle.loads(pickle.dumps(result))
        np.testing.assert_array_equal(restored.params, result.params)
        self.assertEqual(restored.llf, result.llf)


if __name__ == '__main__':
    unittest.main()
