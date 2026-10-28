import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from paneltime import main
from paneltime.api import Model
from paneltime.options import FitOptions
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
        with patch.object(main, "maximize", return_value={"f": -1.0}) as optimize:
            with patch.object(main, "Summary") as summary:
                result = main.execute("y ~ x", self.data, "time", "id", None, options)
        optimize.assert_called_once()
        panel, args = optimize.call_args.args
        self.assertEqual(panel.X.shape, (4, 32, 2))
        self.assertIs(args, panel.args.args_init)
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


if __name__ == "__main__":
    unittest.main()
