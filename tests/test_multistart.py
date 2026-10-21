import unittest
import ast
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import paneltime as pt
from paneltime.main import Input
from paneltime.maximization import maximize
from paneltime.options import FitOptions, OptimizerOptions
from paneltime.processing.panel import Panel


class MultistartTests(unittest.TestCase):
    def make_panel(self, order=(2, 2, 0), garch_order=(2, 2), **kwargs):
        rng = np.random.default_rng(121)
        n, t = 4, 32
        x = rng.normal(size=n*t)
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'x': x, 'y': 0.5 + 0.8*x + rng.normal(size=n*t),
        })
        self.data = data
        options = FitOptions(
            order=order, garch_order=garch_order,
            optimizer=OptimizerOptions(multistart=True), **kwargs,
        ).to_engine_options()
        panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
        panel.init()
        return panel

    def test_option_defaults_and_validation(self):
        self.assertFalse(FitOptions().to_engine_options().multistart)
        positional = OptimizerOptions(1e-6, 200, 1)
        self.assertEqual(positional.accuracy, 1)
        self.assertFalse(positional.multistart)
        self.assertTrue(FitOptions(
            optimizer=OptimizerOptions(multistart=True),
        ).to_engine_options().multistart)
        for invalid in (1, 'yes', None):
            with self.assertRaises(ValueError):
                OptimizerOptions(multistart=invalid).validate()
        options = FitOptions().to_engine_options()
        options.multistart = 1
        with self.assertRaises(TypeError):
            options.validate()

    def test_starts_preserve_levels_and_have_stable_volatility(self):
        panel = self.make_panel()
        args = panel.args.args_init
        before = args.args_v.copy()
        starts = maximize.get_multistart_directions(panel, args)
        self.assertEqual(len(starts), 3)
        np.testing.assert_array_equal(starts[0], before)
        np.testing.assert_array_equal(args.args_v, before)
        for start in starts[1:]:
            for group in ('beta', 'omega'):
                indices = list(args.positions[group])
                np.testing.assert_array_equal(start[indices], before[indices])
            gamma = start[list(args.positions['gamma'])]
            psi = start[list(args.positions['psi'])]
            self.assertTrue(np.all(gamma >= 0) and np.all(psi >= 0))
            self.assertLess(gamma.sum() + psi.sum(), 1)
        np.testing.assert_array_equal(
            starts[1][list(args.positions['rho']) + list(args.positions['lambda'])], 0,
        )

    def test_no_dynamics_deduplicates_and_explicit_start_is_respected(self):
        panel = self.make_panel(order=(0, 0, 0), garch_order=(0, 0))
        self.assertEqual(len(maximize.get_multistart_directions(panel, panel.args.args_init)), 1)
        panel = self.make_panel()
        panel.args.initial_user_defined = True
        starts = maximize.get_multistart_directions(panel, panel.args.args_init)
        self.assertEqual(len(starts), 1)
        np.testing.assert_array_equal(starts[0], panel.args.args_init.args_v)

    def test_ar_only_arch_only_and_egarch(self):
        for order, garch in (((1, 0, 0), (0, 0)), ((0, 1, 0), (0, 1))):
            panel = self.make_panel(order=order, garch_order=garch)
            starts = maximize.get_multistart_directions(panel, panel.args.args_init)
            self.assertGreater(len(starts), 1)
            self.assertTrue(all(np.isfinite(start).all() for start in starts))
        panel = self.make_panel(vol='EGARCH')
        args = panel.args.args_init
        for start in maximize.get_multistart_directions(panel, args):
            for group in ('gamma', 'psi', 'omega'):
                indices = list(args.positions[group])
                np.testing.assert_array_equal(start[indices], args.args_v[indices])

    def test_prefers_converged_fit_and_reports_unsuccessful_starts(self):
        panel = self.make_panel()
        records = {
            0: {'f': -10., 'conv': 1, 'its': 15},
            1: {'f': -5., 'conv': 4, 'its': 20},
            2: {'f': -8., 'conv': 2, 'its': 16},
        }
        with patch.object(maximize, 'maximize_single', return_value=(records, [])):
            result = maximize.maximize(panel, panel.args.args_init, None, 0)
        self.assertIs(result, records[2])
        self.assertEqual(len(result['multistart_results']), 3)
        self.assertEqual(result['multistart_results'][1]['convergence_code'], 4)
        self.assertEqual([row['selected'] for row in result['multistart_results']],
                         [False, False, True])

    def test_no_converged_fit_preserves_failure_and_nonfinite_fits_are_rejected(self):
        panel = self.make_panel()
        records = {0: {'f': -10., 'conv': 3, 'its': 30},
                   1: {'f': np.nan, 'conv': 1, 'its': 15}}
        with patch.object(maximize, 'maximize_single', return_value=(records, [])):
            result = maximize.maximize(panel, panel.args.args_init, None, 0)
        self.assertEqual(result['conv'], 3)
        records[0]['f'] = -np.inf
        with patch.object(maximize, 'maximize_single', return_value=(records, [])):
            with self.assertRaisesRegex(RuntimeError, 'finite log-likelihood'):
                maximize.maximize(panel, panel.args.args_init, None, 0)

    def test_parallel_dispatch_uses_the_same_starts(self):
        panel = self.make_panel()
        args = panel.args.args_init
        mp = SimpleNamespace(
            n_slaves=3,
            eval=lambda tasks: self.assertEqual(len(tasks), 3),
            collect=lambda **kwargs: {0: {'f': -10., 'conv': 1, 'its': 15}},
        )
        starts = maximize.get_multistart_directions(panel, args)
        with patch.object(mp, 'eval') as dispatch:
            maximize.maximize(panel, args, mp, 0)
        tasks = dispatch.call_args.args[0]
        expected = [
            f'maximize.maximize_node(panel, {start.tolist()}, {panel.options.tolerance}, {i}, slave_server)\n'
            for i, start in enumerate(starts)
        ]
        self.assertEqual(tasks, expected)
        for task, start in zip(tasks, starts):
            call = ast.parse(task).body[0].value
            np.testing.assert_array_equal(ast.literal_eval(call.args[1]), start)

    def test_default_dispatch_is_unchanged(self):
        panel = self.make_panel()
        panel.options.multistart = False
        records = {0: {'f': -10., 'conv': 1, 'its': 15}}
        with patch.object(maximize, 'maximize_single', return_value=(records, [])) as run:
            result = maximize.maximize(panel, panel.args.args_init, None, 0)
        run.assert_called_once_with(panel, panel.args.args_init)
        self.assertNotIn('multistart_results', result)

    def test_public_fit_and_constraints(self):
        self.make_panel(order=(1, 1, 0), garch_order=(0, 0))
        data = self.data
        model = pt.Model('y ~ x', data, entity='id', time='time')
        single = model.fit(order=(1, 1, 0), garch_order=(0, 0),
                           constraints={'rho': [0.2]})
        multi = model.fit(order=(1, 1, 0), garch_order=(0, 0),
                          constraints={'rho': [0.2]},
                          optimizer=OptimizerOptions(multistart=True))
        self.assertGreaterEqual(multi.ll.LL, single.ll.LL - 1e-8)
        self.assertEqual(len(multi.optim_result.multistart_results), 3)
        self.assertEqual(single.optim_result.multistart_results, [])
        self.assertAlmostEqual(float(multi.ll.args.args_d['rho'][0, 0]), 0.2)


if __name__ == '__main__':
    unittest.main()
