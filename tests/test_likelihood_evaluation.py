import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import lfilter

import paneltime as pt
from paneltime.likelihood import evaluation


class LikelihoodEvaluationTests(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(267)
        n = 90
        x, z = rng.normal(size=(2, n))
        trend = np.tile(np.arange(30), 3)/30
        y = 1.2+.8*x-.3*z+.6*trend+rng.normal(size=n)
        data = pd.DataFrame(dict(id=np.repeat(np.arange(3), 30),
                                 t=np.tile(np.arange(30), 3),
                                 y=y, x=x, z=z, trend=trend))
        return data[~((data.id == 0) & (data.t >= 25))]

    def test_multiple_regressors_stationary_density_and_all_derivatives(self):
        for difference in (0, 1):
            with self.subTest(difference=difference):
                evaluator = pt.Model("y ~ x + z + trend", self.data(),
                                     entity="id", time="t").prepare_likelihood(
                    pt.FitOptions(order=(2, 1, difference), garch_order=(0, 0), exact_ll=True))
                parameters = evaluator.initial_parameters.copy()
                parameters[list(evaluator.parameter_groups["beta"])] = [1.1, .7, -.2, .5]
                parameters[list(evaluator.parameter_groups["rho"])] = [.3, -.1]
                parameters[list(evaluator.parameter_groups["lambda"])] = [-.2]
                parameters[list(evaluator.parameter_groups["omega"])] = .8
                evaluated = evaluator.evaluate(parameters)
                panel = evaluator._panel
                impulse = np.zeros(32768)
                impulse[0] = 1
                psi = lfilter([1., -.2], [1., -.3, .1], impulse)
                reference = 0.
                for group in range(panel.N):
                    times = np.flatnonzero(panel.included[3][group, :, 0])
                    times = times[times >= difference]
                    residual = (panel.Y[group, times, 0]
                                - panel.X[group, times] @ parameters[
                                    list(evaluator.parameter_groups["beta"])])
                    gamma = [.8*(psi[:len(psi)-k] @ psi[k:]) for k in range(len(times))]
                    covariance = toeplitz(gamma)
                    reference -= .5*(len(times)*np.log(2*np.pi)
                        + np.linalg.slogdet(covariance)[1]
                        + residual @ np.linalg.solve(covariance, residual))
                self.assertAlmostEqual(evaluated.loglike, reference, places=10)
                for i in range(len(parameters)):
                    step = 2e-6
                    up, down = parameters.copy(), parameters.copy()
                    up[i] += step
                    down[i] -= step
                    a, b = evaluator.evaluate(up, 1), evaluator.evaluate(down, 1)
                    self.assertAlmostEqual(evaluated.score[i],
                        (a.loglike-b.loglike)/(2*step), delta=2e-5)
                    np.testing.assert_allclose(evaluated.hessian[:, i],
                        (a.score-b.score)/(2*step), atol=3e-4, rtol=2e-5)

    def test_reuse_is_local_and_results_do_not_mutate_cache(self):
        evaluator = pt.Model("y ~ x + z", self.data().set_index(["id", "t"])).prepare_likelihood(
            pt.FitOptions(order=(1, 1, 0), garch_order=(0, 0), exact_ll=True))
        parameters = evaluator.initial_parameters.copy()
        with patch.object(evaluation, "LL", wraps=evaluation.LL) as likelihood:
            value = evaluator.evaluate(parameters, 0)
            first = evaluator.evaluate(parameters, 1)
            complete = evaluator.evaluate(parameters, 2)
            self.assertEqual(likelihood.call_count, 1)
            score, hessian = complete.score.copy(), complete.hessian.copy()
            complete.score[:] = 0
            complete.hessian[:] = 0
            repeated = evaluator.evaluate(parameters)
            np.testing.assert_array_equal(repeated.score, score)
            np.testing.assert_array_equal(repeated.hessian, hessian)
            parameters[0] += .2
            changed = evaluator.evaluate(parameters)
            self.assertEqual(likelihood.call_count, 2)
            self.assertNotEqual(value.loglike, changed.loglike)
            self.assertIsNone(value.score)
            self.assertIsNone(first.hessian)

    def test_invalid_parameters_raise_and_valid_evaluation_remains_usable(self):
        evaluator = pt.Model("y ~ x", self.data(), entity="id", time="t").prepare_likelihood(
            pt.FitOptions(order=(1, 1, 0), garch_order=(0, 0), exact_ll=True))
        parameters = evaluator.initial_parameters.copy()
        before = evaluator.evaluate(parameters)
        for invalid in (parameters[:-1], np.full_like(parameters, np.nan)):
            with self.assertRaisesRegex(ValueError, "finite vector"):
                evaluator.evaluate(invalid)
        invalid = parameters.copy()
        invalid[list(evaluator.parameter_groups["rho"])] = 1.1
        with self.assertRaisesRegex(ValueError, "Invalid likelihood"):
            evaluator.evaluate(invalid)
        with self.assertRaisesRegex(ValueError, "derivative_order"):
            evaluator.evaluate(parameters, 3)
        self.assertEqual(evaluator.evaluate(parameters).loglike, before.loglike)

    def test_explicitly_fixed_full_vector_is_evaluated_without_optimization(self):
        data = self.data()
        for exact in (False, True):
            model = pt.Model("y ~ x + z + trend", data, entity="id", time="t")
            options = pt.FitOptions(order=(1, 1, 0), garch_order=(0, 0), exact_ll=exact)
            evaluator = model.prepare_likelihood(options)
            parameters = evaluator.initial_parameters.copy()
            fixed = dict(beta=[1.1, .7, -.2, .5], rho=.3, **{"lambda": -.2}, omega=.8)
            for name, value in fixed.items():
                parameters[list(evaluator.parameter_groups[name])] = value
            expected = evaluator.evaluate(parameters, 0)
            with patch("paneltime.maximization.dfpmax._iterate",
                       side_effect=AssertionError("Unexpected optimization")):
                result = model.fit(order=(1, 1, 0), garch_order=(0, 0),
                                   exact_ll=exact, constraints=fixed)
            np.testing.assert_allclose(result.params.to_numpy(), parameters, rtol=0, atol=1e-14)
            self.assertAlmostEqual(result.llf, expected.loglike, places=10)
            self.assertEqual(result.general.its, 0)
