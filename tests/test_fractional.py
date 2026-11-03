import pickle
import unittest
import importlib.util

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import paneltime as pt
from paneltime.likelihood.calculus import gradient, hessian
from paneltime.likelihood.fractional import FractionalFilter
from paneltime.likelihood.main import LL, pred_u
from paneltime.likelihood.models import Normal, Exponential, Hyperbolic
from paneltime.main import Input
from paneltime.options import FitOptions
from paneltime.processing.panel import Panel
from paneltime.maximization.computation import Computation
from tests.test_likelihood_models import NumericNormal


class FractionalTests(unittest.TestCase):
    @staticmethod
    def data(n=4, t=24, seed=31):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'x': rng.normal(size=n*t), 'y': rng.normal(size=n*t),
        })

    def panel(self, d='estimate', order=(1, 1, 0), garch=(1, 1),
              effects=None, vol='GARCH', h_function=None, likelihood=None, data=None):
        data = self.data() if data is None else data
        options = FitOptions(
            order=order, garch_order=garch, fractional_d=d,
            effects=effects or pt.Effects(), vol=vol, h_function=h_function,
            likelihood=likelihood,
        ).to_engine_options()
        panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
        panel.init()
        return panel

    def test_weights_and_derivatives_at_zero_and_nonzero_d(self):
        for d in (0., .3, -.3):
            f = FractionalFilter(d, 40)
            self.assertEqual(f.weights[0], 1)
            self.assertEqual(f.weights[1], -d)
            self.assertAlmostEqual(f.weights[2], d*(d-1)/2)
            step = 1e-5
            up, down = FractionalFilter(d+step, 40), FractionalFilter(d-step, 40)
            np.testing.assert_allclose(f.first, (up.weights-down.weights)/(2*step), atol=1e-10)
            np.testing.assert_allclose(f.second, (up.first-down.first)/(2*step), atol=1e-10)
        np.testing.assert_array_equal(FractionalFilter(0., 40).weights[1:], 0)

    def test_filter_matches_independent_convolution_and_never_joins_entities(self):
        values = np.random.default_rng(1).normal(size=(3, 18, 2))
        f = FractionalFilter(.25, 19)
        expected = np.empty_like(values)
        for group in range(3):
            for variable in range(2):
                expected[group, :, variable] = np.convolve(
                    values[group, :, variable], f.weights)[:18]
        np.testing.assert_allclose(f.apply(values), expected, atol=1e-14)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(f)).apply(values), expected)
        for model in (Normal, Exponential, Hyperbolic, NumericNormal):
            self.assertIsInstance(model.fractional_filter(.2, 20), FractionalFilter)

    def test_d_zero_is_exact_existing_likelihood_and_derivative_path(self):
        for garch in ((0, 0), (1, 1)):
            ordinary = self.panel(None, garch=garch)
            fractional = self.panel(0., garch=garch)
            args = ordinary.args.args_init.args_v
            a, b = LL(args, ordinary), LL(args, fractional)
            self.assertEqual(a.LL, b.LL)
            ga, gb = gradient(ordinary), gradient(fractional)
            np.testing.assert_array_equal(ga.get(a)[0], gb.get(b)[0])
            np.testing.assert_array_equal(
                hessian(ordinary, ga).get(a, *a.llfunc.hessian()),
                hessian(fractional, gb).get(b, *b.llfunc.hessian()))

    def check_derivatives(self, panel):
        args = panel.args.args_init.args_v.copy()
        for name, value in (('rho', .2), ('lambda', -.1), ('gamma', .3),
                            ('psi', .1), ('omega', .7), ('d', .23), ('z', .2)):
            if name in panel.args.positions:
                args[list(panel.args.positions[name])] = value

        def evaluate(a):
            ll = LL(a, panel)
            g = gradient(panel)
            score, scores = g.get(ll)
            return ll, g, score, scores

        ll, g, score, scores = evaluate(args)
        H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
        self.assertEqual(H.shape, (len(args), len(args)))
        np.testing.assert_allclose(scores.sum(axis=(0, 1)), score)
        step = 2e-6
        for i in range(len(args)):
            up, down = args.copy(), args.copy()
            up[i] += step
            down[i] -= step
            up_ll, _, up_score, _ = evaluate(up)
            down_ll, _, down_score, _ = evaluate(down)
            self.assertAlmostEqual(score[i], (up_ll.LL-down_ll.LL)/(2*step), delta=2e-5)
            np.testing.assert_allclose(H[:, i], (up_score-down_score)/(2*step),
                                       atol=3e-4, rtol=2e-5)

    def test_full_analytic_derivatives(self):
        for d in ('estimate', .2):
            for vol, effects, likelihood in (
                ('GARCH', pt.Effects(), None),
                ('EGARCH', pt.Effects(), None),
                ('GARCH', pt.Effects(group='fixed', time='fixed'), None),
                ('GARCH', pt.Effects(group='random', time='random'), None),
                ('GARCH', pt.Effects(), NumericNormal),
            ):
                with self.subTest(d=d, vol=vol, effects=effects, model=likelihood):
                    self.check_derivatives(self.panel(
                        d, vol=vol, effects=effects, likelihood=likelihood))

    @unittest.skipUnless(importlib.util.find_spec('sympy'), 'SymPy is optional')
    def test_shape_cross_derivatives(self):
        import sympy
        e, z = sympy.symbols('e z', real=True)
        self.check_derivatives(self.panel(h_function=e**2 + z))

    def test_higher_orders_and_no_arma_or_garch_derivatives(self):
        for order, garch in (((2, 2, 0), (2, 2)), ((0, 0, 0), (0, 0)),
                             ((1, 0, 0), (0, 0)), ((0, 1, 0), (1, 1))):
            with self.subTest(order=order, garch=garch):
                self.check_derivatives(self.panel(order=order, garch=garch))

    def test_negative_and_zero_d_derivatives(self):
        for value in (0., -.2):
            panel = self.panel()
            args = panel.args.args_init.args_v.copy()
            args[-1] = value
            ll = LL(args, panel)
            g = gradient(panel)
            g.get(ll)
            H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
            step = 1e-6
            up, down = args.copy(), args.copy()
            up[-1] += step
            down[-1] -= step
            numeric = (gradient(panel).get(LL(up, panel))[0]
                       - gradient(panel).get(LL(down, panel))[0]) / (2*step)
            np.testing.assert_allclose(H[:, -1], numeric, rtol=2e-5, atol=1e-4)

    def test_ragged_panel_derivatives_and_forecast_inversion(self):
        data = self.data()
        data = data[~((data.id == 0) & (data.time >= 18))]
        panel = self.panel(data=data)
        self.check_derivatives(panel)
        ll = LL(panel.args.args_init.args_v, panel)
        predicted = ll.predict(panel)
        self.assertFalse(predicted.empty)
        filtered = pred_u(ll.u_fractional, ll.e, ll.args.args_d['rho'],
                          ll.args.args_d['lambda'], panel)
        inverted = ll.fractional.forecast(filtered, ll.u, panel.T_arr)
        np.testing.assert_allclose(ll.u_pred, inverted, equal_nan=True)
        for group, length in enumerate(panel.T_arr[:, 0]):
            extended = np.concatenate((ll.u[group:group+1, :length],
                                       inverted[group:group+1, None, :]), axis=1)
            np.testing.assert_allclose(
                ll.fractional.apply(extended)[:, -1], filtered[group:group+1], atol=1e-12)

    def test_rejects_invalid_d_integer_differencing_and_gaps(self):
        for d in (True, np.bool_(False), np.nan, np.inf, -.5, .5, 'invalid'):
            with self.subTest(d=d), self.assertRaises(ValueError):
                FitOptions(fractional_d=d).to_engine_options()
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.panel(.2, order=(1, 1, 1))
        data = self.data()
        for dates in (data[~((data.id == 0) & (data.time == 8))],
                      data.assign(time=data.time**2)):
            with self.assertRaisesRegex(ValueError, 'regular'):
                self.panel(.2, data=dates)

    def test_datetime_frequency_validation(self):
        data = self.data()
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        data['time'] = dates[data.time]
        self.assertIsNotNone(self.panel(.2, data=data))
        with self.assertRaisesRegex(ValueError, 'regular'):
            self.panel(.2, data=data.drop(index=8))

    def test_constraints_preserve_stationary_bounds(self):
        panel = self.panel(order=(0, 0, 0), garch=(0, 0))
        for restriction in ((-.2, .3), (None, .8), .25):
            with self.subTest(restriction=restriction):
                panel.args.user_constraints = {'d': restriction}
                computation = Computation(panel.args.args_init.args_v, panel, 1e-4, 1e-8, False)
                constraint = computation.constr[panel.args.positions['d'][0]]
                if isinstance(restriction, tuple):
                    self.assertGreater(constraint.min, -.5)
                    self.assertLess(constraint.max, .5)
                    self.assertLessEqual(constraint.min, 0)
                    self.assertGreaterEqual(constraint.max, .3)
                else:
                    self.assertEqual(constraint.value, restriction)
        for restriction in (.6, (-.9, -.6), (.6, .9), (np.nan, .3), (-.2, np.nan)):
            panel.args.user_constraints = {'d': restriction}
            with self.subTest(restriction=restriction), self.assertRaises(ValueError):
                Computation(panel.args.args_init.args_v, panel, 1e-4, 1e-8, False)

    def test_fixed_and_constrained_estimated_fits_agree_without_garch(self):
        data = self.data(n=1, t=100)
        model = pt.Model('y ~ x', data, entity='id', time='time')
        kwargs = dict(order=(0, 0, 0), garch_order=(0, 0),
                      optimizer=pt.OptimizerOptions(tolerance=1e-8))
        fixed = model.fit(fractional_d=.2, **kwargs)
        constrained = model.fit(fractional_d='estimate', constraints={'d': .2}, **kwargs)
        self.assertTrue(fixed.converged)
        self.assertTrue(constrained.converged)
        self.assertAlmostEqual(fixed.ll.LL, constrained.ll.LL, delta=1e-5)
        self.assertEqual(fixed.fractional_d, .2)
        self.assertEqual(constrained.fractional_d, .2)
        self.assertNotIn('d', fixed.params.index)
        self.assertIn('d', constrained.params.index)
        np.testing.assert_allclose(fixed.forecast(), constrained.forecast(), atol=1e-5)
        restored = pickle.loads(pickle.dumps(fixed))
        np.testing.assert_array_equal(restored.forecast(), fixed.forecast())

    def test_legacy_custom_models_use_shared_filter_fallback(self):
        class LegacyNormal:
            def __new__(cls, *args):
                return Normal(*args)
        panel = self.panel(.2, likelihood=LegacyNormal)
        ll = LL(panel.args.args_init.args_v, panel)
        self.assertIsNotNone(ll.LL)
        self.assertIsInstance(ll.fractional, FractionalFilter)

    def test_joint_arfima_garch_fit_inference_and_forecasts(self):
        rng = np.random.default_rng(42)
        n, t = 3, 250
        innovations = np.zeros((n, t, 1))
        variance = np.ones_like(innovations)
        residuals = np.zeros_like(innovations)
        for date in range(t):
            if date:
                variance[:, date] = (.2 + .65*variance[:, date-1]
                                     + .15*innovations[:, date-1]**2)
            innovations[:, date, 0] = rng.normal(size=n)*np.sqrt(variance[:, date, 0])
            residuals[:, date] = innovations[:, date] + (
                .25*residuals[:, date-1] if date else 0)
        residuals = FractionalFilter(-.25, t).apply(residuals)
        x = rng.normal(size=(n, t, 1))
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'x': x.ravel(), 'y': (.4 + .8*x + residuals).ravel(),
        })
        result = pt.Model('y ~ x', data, entity='id', time='time').fit(
            order=(1, 0, 0), garch_order=(1, 1), fractional_d='estimate',
            optimizer=pt.OptimizerOptions(tolerance=1e-7))
        self.assertTrue(result.converged)
        self.assertTrue(-.5 < result.fractional_d < .5)
        self.assertTrue(np.isfinite(result.bse.to_numpy()).all())
        self.assertTrue(np.isfinite(result.bse_at(result.params).to_numpy()).all())
        self.assertEqual(result.forecast().shape, (n,))
        self.assertTrue(np.isfinite(result.forecast()).all())

    def test_estimation_matches_independent_profile_likelihood(self):
        rng = np.random.default_rng(4)
        t, d = 500, .3
        # Generate the inverse filter independently from its binomial recurrence.
        inverse = np.ones(t)
        for lag in range(1, t):
            inverse[lag] = inverse[lag-1]*(lag-1+d)/lag
        y = np.convolve(rng.normal(size=t), inverse)[:t]
        data = pd.DataFrame({'y': y, 'time': np.arange(t)})
        result = pt.Model('y ~ 1', data, time='time').fit(
            order=(0, 0, 0), garch_order=(0, 0), fractional_d='estimate',
            optimizer=pt.OptimizerOptions(tolerance=1e-8))
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.bse.to_numpy()).all())

        def profile(d):
            weights = np.ones(t)
            for lag in range(1, t):
                weights[lag] = weights[lag-1]*(lag-1-d)/lag
            filtered_y = np.convolve(y, weights)[:t]
            filtered_constant = np.cumsum(weights)
            beta = filtered_constant @ filtered_y / (filtered_constant @ filtered_constant)
            residual = filtered_y-beta*filtered_constant
            return t/2*np.log(residual @ residual/t)

        reference = minimize_scalar(profile, bounds=(-.49, .49), method='bounded')
        self.assertAlmostEqual(result.fractional_d, reference.x, delta=2e-4)
        self.assertAlmostEqual(result.fractional_d, d, delta=.12)
        self.assertTrue(np.isfinite(result.forecast()).all())


if __name__ == '__main__':
    unittest.main()
