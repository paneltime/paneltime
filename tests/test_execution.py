import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paneltime import main
from paneltime.api import Model
from paneltime.options import FitOptions, OptimizerOptions
from paneltime.output import output


class LocalExecutionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=128)
        self.data = pd.DataFrame({
            "id": np.repeat(np.arange(4), 32),
            "time": np.tile(np.arange(32), 4),
            "x": x,
            "y": 0.5 + 0.8 * x + rng.normal(size=128),
        })

    def test_execute_calls_optimizer_directly_with_initialized_panel(self):
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
        original = self.data.copy(deep=True)
        with patch.object(main, "dfpmax", return_value={"f": -1.0}) as optimize:
            with patch.object(main, "Summary") as summary:
                result = main.execute("y ~ x", self.data, "time", "id", None, options)
        optimize.assert_called_once()
        args, panel = optimize.call_args.args
        self.assertEqual(panel.X.shape, (4, 32, 2))
        self.assertIs(args, panel.args.args_init.args_v)
        self.assertIs(summary.call_args.args[1], panel)
        self.assertIs(result, summary.return_value)
        pd.testing.assert_frame_equal(self.data, original)

    def test_suppressed_fit_builds_final_statistics_once_without_progress_state(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with patch.object(output, "Statistics", wraps=output.Statistics) as statistics:
                result = Model("y ~ x", self.data, entity="id", time="time").fit(
                    order=(0, 0, 0), garch_order=(0, 0),
                )
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(statistics.call_count, 1)
        self.assertFalse(hasattr(result.optim_result, "comm"))
        self.assertFalse(hasattr(result.optim_result, "multistart_results"))
        self.assertFalse(hasattr(result.panel.input, "n_nodes"))
        self.assertTrue(np.isfinite(result.llf))
        self.assertTrue(np.isfinite(result.bse).all())
        self.assertTrue(np.isfinite(result.robust_bse).all())
        self.assertIn("Statistics:", str(result))
        self.assertIn("Diagnostics:", result.diagnostics())
        self.assertEqual(result.forecast().shape, (4,))

    def test_unsuppressed_fit_reports_final_status_without_node_messages(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            Model("y ~ x", self.data, entity="id", time="time").fit(
                order=(0, 0, 0), garch_order=(0, 0), suppress_output=False,
            )
        self.assertIn("Iterations:", stdout.getvalue())
        self.assertIn("LL:", stdout.getvalue())
        self.assertNotIn("node:", stdout.getvalue())

    def test_removed_progress_arguments_are_rejected(self):
        options = FitOptions().to_engine_options()
        with self.assertRaisesRegex(TypeError, "window"):
            main.execute("y ~ x", self.data, "time", "id", None, options, window=object())

    def test_removed_multistart_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "multistart"):
            OptimizerOptions(multistart=True)
        options = FitOptions().to_engine_options()
        self.assertFalse(hasattr(options, "multistart"))
        with self.assertRaisesRegex(AttributeError, "multistart"):
            options.multistart = True

    def test_optimizer_option_positions_and_validation_are_preserved(self):
        options = OptimizerOptions(1e-6, 200, 1)
        self.assertEqual(options.tolerance, 1e-6)
        self.assertEqual(options.max_iterations, 200)
        self.assertEqual(options.accuracy, 1)
        options.validate()
        with self.assertRaises(ValueError):
            OptimizerOptions(tolerance=-1).validate()

    def test_optimizer_errors_are_not_hidden(self):
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
        with patch.object(main, "dfpmax", side_effect=RuntimeError("Invalid likelihood")):
            with self.assertRaisesRegex(RuntimeError, "Invalid likelihood"):
                main.execute("y ~ x", self.data, "time", "id", None, options)

    def test_single_start_preserves_parameter_constraints(self):
        result = Model("y ~ x", self.data, entity="id", time="time").fit(
            order=(1, 1, 0), garch_order=(0, 0), constraints={"rho": [0.2]},
        )
        self.assertAlmostEqual(float(result.ll.args.args_d["rho"][0, 0]), 0.2)
        self.assertTrue(np.isfinite(result.llf))


if __name__ == "__main__":
    unittest.main()
