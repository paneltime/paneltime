import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import paneltime as pt
import paneltime.api as api
from paneltime.cfunctions import _library_name
from paneltime.likelihood.symbolic import build_h_function
from paneltime.options import FitOptions
from paneltime.processing.model_parser import get_names, pool_func

try:
    import sympy as sp
except ImportError:
    sp = None


class PackageWorkflowTests(unittest.TestCase):
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

    def test_paneltime_mp_is_only_required_when_enabled(self):
        data = pd.DataFrame({'id': [1, 1, 2, 2], 't': [1, 2, 1, 2], 'y': [1.0, 2.0, 3.0, 4.0]})

        with patch.object(api, 'MP', None), patch.dict('sys.modules', {'paneltime_mp': None}):
            self.assertIsInstance(api.Model('y', data, entity='id', time='t'), api.Model)
            with self.assertRaisesRegex(ImportError, 'requires the optional paneltime_mp package'):
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
            order=(0, 0, 0), garch_order=(0, 0), suppress_output=True
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