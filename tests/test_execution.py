import contextlib
import io
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from paneltime import main
from paneltime.api import Model
from paneltime.options import FitOptions, OptimizerOptions
from paneltime.output import output
from paneltime.processing import initreg


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
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0), constraints={}).to_engine_options()
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
                    order=(0, 0, 0), garch_order=(0, 0), constraints={},
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
                order=(0, 0, 0), garch_order=(0, 0), constraints={}, suppress_output=False,
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
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0), constraints={}).to_engine_options()
        with patch.object(main, "dfpmax", side_effect=RuntimeError("Invalid likelihood")):
            with self.assertRaisesRegex(RuntimeError, "Invalid likelihood"):
                main.execute("y ~ x", self.data, "time", "id", None, options)

    def test_static_start_reuses_panel_and_runs_shared_optimizer(self):
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
        with patch.object(main, "Panel", wraps=main.Panel) as create_panel:
            with patch.object(initreg, "static_start",
                              wraps=initreg.static_start) as stationary:
                with patch.object(main, "dfpmax", wraps=main.dfpmax) as optimize:
                    result = main.execute("y ~ x", self.data, "time", "id", None, options)
        create_panel.assert_called_once()
        stationary.assert_called_once_with(result.panel)
        optimize.assert_called_once()
        self.assertEqual(result.general.its, 0)
        self.assertFalse(hasattr(result, "closed_form"))

    def test_static_parameters_are_ready_when_panel_initialization_finishes(self):
        from paneltime.likelihood.main import LL
        from paneltime.random_effects.static import fit_group_effects_stationary
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
        panel = main.Panel(main.Input(self.data, "y ~ x", "id", "time", options, None, None), options)
        with patch.object(initreg.arma_start, "working_regression",
                          side_effect=AssertionError("Redundant starting regression")):
            panel.init()
        expected = fit_group_effects_stationary(panel).beta.to_numpy()
        args = np.asarray(panel.args.args_init.args_v)
        np.testing.assert_allclose(args[list(panel.args.positions["beta"])], expected)
        likelihood = LL(args, panel)
        omega = float(args[panel.args.positions["omega"][0]])
        self.assertAlmostEqual(omega, np.sum(likelihood.e2)/panel.NT)
        self.assertAlmostEqual(panel.args.init_var, omega)

    def test_ineligible_models_skip_static_initialization(self):
        for fit_options in (
            FitOptions(order=(0, 0, 0), garch_order=(0, 0), constraints={}),
            FitOptions(order=(0, 0, 0), garch_order=(1, 1)),
        ):
            with self.subTest(order=fit_options.garch_order, constraints=fit_options.constraints):
                options = fit_options.to_engine_options()
                panel = main.Panel(main.Input(self.data, "y ~ x", "id", "time", options, None, None), options)
                with patch.object(initreg, "static_start", side_effect=AssertionError("Unexpected static start")):
                    panel.init()

    def test_static_numerical_errors_are_not_retried_with_optimizer(self):
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
        with patch.object(initreg, "static_start",
                          side_effect=np.linalg.LinAlgError("Static fit failed")):
            with patch.object(main, "dfpmax") as optimize:
                with self.assertRaisesRegex(np.linalg.LinAlgError, "Static fit failed"):
                    main.execute("y ~ x", self.data, "time", "id", None, options)
        optimize.assert_not_called()

    def test_optimal_static_start_does_not_run_line_search(self):
        from paneltime.maximization import linesearch
        with patch.object(linesearch, "LineSearch", side_effect=AssertionError("Unnecessary iteration")):
            result = Model("y ~ x", self.data, entity="id", time="time").fit(
                order=(0, 0, 0), garch_order=(0, 0), suppress_output=True,
            )
        self.assertEqual(result.optim_result.its, 0)
        self.assertTrue(result.converged)

    def test_explicit_engine_start_is_not_replaced_by_static_estimator(self):
        options = FitOptions(order=(0, 0, 0), garch_order=(0, 0)).to_engine_options()
        panel = main.Panel(main.Input(self.data, "y ~ x", "id", "time", options, None, None), options)
        panel.init()
        options.arguments = panel.args.args_init.args_v.copy()
        with patch.object(initreg, "static_start", side_effect=AssertionError("Start replaced")):
            with patch.object(main, "dfpmax", return_value={"f": -1.0}) as optimize:
                with patch.object(main, "Summary"):
                    main.execute("y ~ x", self.data, "time", "id", None, options)
        np.testing.assert_array_equal(optimize.call_args.args[0], options.arguments)

    def test_converged_requires_successful_optimizer_status(self):
        from paneltime.api import Summary
        result = Summary.__new__(Summary)
        result._summary = SimpleNamespace(general=SimpleNamespace(converged=0))
        for code in range(7):
            result._summary.general.converged = code
            self.assertEqual(result.converged, code in (1, 2))

    def test_reporting_errors_propagate_even_with_suppressed_output(self):
        with patch.object(output.RegTableObj, "set", side_effect=TypeError("Broken statistics")):
            with self.assertRaisesRegex(TypeError, "Broken statistics"):
                Model("y ~ x", self.data, entity="id", time="time").fit(
                    order=(0, 0, 0), garch_order=(0, 0), suppress_output=True,
                )

    def test_fit_preserves_callers_warning_policy(self):
        initialize = main.Panel.init

        def initialize_with_warning(panel):
            warnings.warn("Caller controls this warning", UserWarning)
            return initialize(panel)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            with patch.object(main.Panel, "init", initialize_with_warning):
                result = Model("y ~ x", self.data, entity="id", time="time").fit(
                    order=(0, 0, 0), garch_order=(0, 0), suppress_output=True,
                )
        self.assertTrue(np.isfinite(result.llf))
        self.assertTrue(any(str(w.message) == "Caller controls this warning" for w in caught))

    def test_invalid_constraints_are_not_silently_discarded(self):
        from paneltime.processing.arguments import arguments
        args = arguments.__new__(arguments)
        args.caption_d = {"beta": [], "omega": []}
        panel = SimpleNamespace(z_active=False)
        for value, error in (("{", SyntaxError), ("not_defined", NameError),
                             ("[]", TypeError), ({"bogus": 0.0}, ValueError)):
            with self.subTest(value=value):
                panel.options = SimpleNamespace(user_constraints=value)
                with self.assertRaises(error):
                    args.get_user_constraints(panel)
        supplied = {"beta": [1.0]}
        panel.options = SimpleNamespace(user_constraints=supplied)
        args.get_user_constraints(panel)
        self.assertEqual(args.user_constraints, supplied)
        self.assertIsNot(args.user_constraints, supplied)

    def test_hessian_inversion_only_recovers_from_singularity(self):
        from paneltime.maximization.computation import hess_inv
        np.testing.assert_allclose(hess_inv(np.diag([1., 0.]), None), np.diag([1., 0.]))
        with patch("numpy.linalg.inv", side_effect=TypeError("Invalid Hessian")):
            with self.assertRaisesRegex(TypeError, "Invalid Hessian"):
                hess_inv(np.eye(2), None)
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("Singular")):
            with patch("numpy.linalg.pinv", side_effect=np.linalg.LinAlgError("SVD failed")):
                with self.assertRaisesRegex(np.linalg.LinAlgError, "SVD failed"):
                    hess_inv(np.eye(2), None)

    def test_covariance_expansion_does_not_hide_shape_errors(self):
        idx = np.array([True, False, True])
        block = np.array([[1., .2], [.2, 2.]])
        expanded = output.expand_x(block, idx, matrix=True)
        np.testing.assert_allclose(expanded[np.ix_(idx, idx)], block)
        self.assertTrue(np.isnan(expanded[1]).all())
        with self.assertRaises(ValueError):
            output.expand_x(np.ones((3, 3)), idx, matrix=True)

    def test_ols_does_not_return_zero_estimates_after_solver_failure(self):
        from paneltime.output.stat_functions import OLS
        panel = SimpleNamespace(included=[None, None, None, np.ones((1, 4, 1))], NT=4)
        with patch("paneltime.output.stat_functions.fn.solve",
                   side_effect=np.linalg.LinAlgError("Solver failed")):
            with self.assertRaisesRegex(np.linalg.LinAlgError, "Solver failed"):
                OLS(panel, np.ones((1, 4, 1)), np.ones((1, 4, 1)))

    def test_date_means_does_not_hide_invalid_cached_indices(self):
        from paneltime.random_effects import dynamic
        values = np.ones((2, 3, 1))
        panel = SimpleNamespace(
            date_count=np.ones((3, 1, 1)), n_dates=3,
            included=[None, None, None, np.ones_like(values)],
            date_flat_index=np.array([7]), date_starts=np.array([0]),
        )
        with self.assertRaises(IndexError):
            dynamic.mean_time(panel, values)

    def test_single_start_preserves_parameter_constraints(self):
        result = Model("y ~ x", self.data, entity="id", time="time").fit(
            order=(1, 1, 0), garch_order=(0, 0), constraints={"rho": [0.2]},
        )
        self.assertAlmostEqual(float(result.ll.args.args_d["rho"][0, 0]), 0.2)
        self.assertTrue(np.isfinite(result.llf))


if __name__ == "__main__":
    unittest.main()
