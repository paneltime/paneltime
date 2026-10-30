import unittest

import numpy as np
import pandas as pd

from paneltime.likelihood.calculus import gradient
from paneltime.likelihood.main import LL
from paneltime.main import Input
from paneltime.options import Effects, FitOptions
from paneltime.processing.panel import Panel
from re_likelihood_reference import (
    benchmark_case, covariance, expected_exact_loglike, expected_quasi_loglike,
    gaussian_loglike, quasi_demeaning, transformed_variance, woodbury_loglike,
)


class RELikelihoodBenchmarkTests(unittest.TestCase):
    def test_dense_and_integrated_innovations_likelihoods_agree(self):
        n, t = 3, 7
        residual = np.sin(np.arange(n*t)*0.7) + np.arange(n*t)/(n*t)
        for rho in (-0.4, 0., 0.65):
            for vg, vt in ((0., 0.), (1., 0.), (0., 0.7), (1., 0.7)):
                cov = covariance(n, t, rho, 2., vg, vt)
                dense = gaussian_loglike(residual, cov)
                integrated = woodbury_loglike(residual, n, t, rho, 2., vg, vt)
                self.assertAlmostEqual(dense, integrated, places=10)

    def test_covariance_validation(self):
        for arguments in ((0, 8, .5, 2, 1, 1), (4, 1, .5, 2, 1, 1),
                          (4, 8, 1., 2, 1, 1), (4, 8, .5, 0, 1, 1),
                          (4, 8, .5, 2, -1, 1), (4, 8, np.nan, 2, 1, 1)):
            with self.assertRaises(ValueError):
                covariance(*arguments)

    def test_covariance_trace_is_an_exact_expectation(self):
        n, t = 2, 4
        true_cov = covariance(n, t, .5, 2., 1., .7)
        candidate = covariance(n, t, -.2, 1.5, .4, .3)
        # These columns have exactly the desired second moment; no random draws.
        columns = np.sqrt(n*t)*np.linalg.cholesky(true_cov)
        values = [gaussian_loglike(columns[:, j], candidate) for j in range(n*t)]
        self.assertAlmostEqual(np.mean(values), expected_exact_loglike(candidate, true_cov), places=10)
        a = quasi_demeaning(n, t, .6, .3)
        b, variance = transformed_variance(n, t, .2, 1.5, 2.)
        w = a@b
        values = [-.5*(n*t*np.log(2*np.pi) + np.log(variance).sum()
                       + ((w@columns[:, j])**2/variance).sum()) for j in range(n*t)]
        self.assertAlmostEqual(
            np.mean(values), expected_quasi_loglike(n, t, .2, 1.5, 2., a, true_cov), places=10,
        )

    def test_no_effects_whitens_and_has_zero_expected_score(self):
        for rho in (0., .5):
            result = benchmark_case(rho=rho, group_variance=0., time_variance=0.)
            self.assertLess(result['whitened_covariance_max_error'], 1e-12)
            np.testing.assert_allclose(result['exact_expected_score_rho_log_variance'], 0, atol=1e-7)
            np.testing.assert_allclose(result['frozen_quasi_expected_score_rho_log_variance'], 0, atol=1e-7)
            self.assertAlmostEqual(result['expected_exact_loglike'], result['expected_quasi_loglike'], places=10)

    def test_re_whitening_and_score_mismatch_is_not_a_constant_offset(self):
        for rho in (0., .5):
            result = benchmark_case(rho=rho)
            np.testing.assert_allclose(result['exact_expected_score_rho_log_variance'], 0, atol=1e-7)
            self.assertGreater(result['whitened_covariance_max_error'], .1)
            self.assertGreater(np.linalg.norm(result['frozen_quasi_expected_score_rho_log_variance']), .1)
        result = benchmark_case(time_variance=0.)
        self.assertGreater(abs(result['frozen_quasi_expected_score_rho_log_variance'][0]), .1)

    def test_grand_mean_correction_is_sufficient_only_in_the_iid_special_case(self):
        for vg, vt in ((1., 0.), (0., 1.), (1., 1.)):
            iid = benchmark_case(rho=0., group_variance=vg, time_variance=vt)
            ar = benchmark_case(rho=.5, group_variance=vg, time_variance=vt)
            self.assertLess(iid['iid_grand_adjusted_whitening_max_error'], 1e-12)
            self.assertGreater(ar['iid_grand_adjusted_whitening_max_error'], .01)

    def test_reference_transform_matches_actual_package_likelihood(self):
        n, t, rho, sigma2 = 4, 9, .4, 2.
        grid = np.arange(n*t).reshape(n, t)
        y = 2*np.arange(n)[:, None] + np.sin(grid*1.3) + np.cos(np.arange(t)*.8)
        data = pd.DataFrame({
            'id': np.repeat(np.arange(n), t), 'time': np.tile(np.arange(t), n),
            'y': y.reshape(-1),
            'x': np.cos(grid.reshape(-1)*.45),
        })
        options = FitOptions(
            order=(1, 0, 0), garch_order=(0, 0),
            effects=Effects(group='random', time='random'),
        ).to_engine_options()
        panel = Panel(Input(data, 'y ~ x', 'id', 'time', options, None, None), options)
        panel.init()
        # Conditional likelihood: the first innovation has variance sigma2 too.
        first_variance = sigma2
        args = panel.args.args_init.args_v.copy()
        args[list(panel.args.positions['beta'])] = 0.
        args[list(panel.args.positions['rho'])] = rho
        args[list(panel.args.positions['omega'])] = sigma2
        ll = LL(args, panel)
        theta_g = float(ll.re_obj_i.theta[0, 0, 0])
        theta_t = float(ll.re_obj_t.theta[0, 0, 0])
        a = quasi_demeaning(n, t, theta_g, theta_t)
        b, variance = transformed_variance(n, t, rho, sigma2, first_variance)
        u = panel.Y.reshape(-1)
        # Effects are removed after the ARMA filter: e = a b u, with theta estimated from b u.
        e = a@b@u
        np.testing.assert_allclose(ll.e.reshape(-1), b@u, atol=1e-12)
        np.testing.assert_allclose(ll.e_RE.reshape(-1), e, atol=1e-12)
        np.testing.assert_allclose(ll.var.reshape(-1), variance, atol=1e-12)
        expected = -.5*(n*t*np.log(2*np.pi) + np.log(variance).sum()
                        + ((e*e + 1e-8)/variance).sum())
        self.assertAlmostEqual(ll.LL, expected, places=10)
        self.assertGreater(abs(np.linalg.slogdet(a)[1]), 1.)
        # The rho score includes the dependence of theta on rho.
        score = gradient(panel).get(ll)[0]
        step = 1e-5
        j = list(panel.args.positions['rho'])[0]
        values = []
        for sign in (-1, 1):
            changed = args.copy()
            changed[j] += sign*step
            values.append(LL(changed, panel).LL)
        self.assertAlmostEqual(score[j], (values[1]-values[0])/(2*step), places=6)


if __name__ == '__main__':
    unittest.main()
