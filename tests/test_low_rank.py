import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_triangular
from scipy.signal import lfilter

from paneltime.likelihood.calculus import gradient, hessian
from paneltime.likelihood.covariance import CovarianceState, LowRankGaussianCovariance
from paneltime.likelihood.low_rank import LowRankBlock, StationaryLoading
from paneltime.likelihood.main import LL
from paneltime.likelihood import low_rank
from tests import test_covariance as fixtures


class LowRankTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CovarianceTests()

    @staticmethod
    def reference(ar, ma, length):
        impulse = np.zeros(32768)
        impulse[0] = 1
        response = lfilter(np.r_[1., ma], np.r_[1., -np.asarray(ar)], impulse)
        return toeplitz([response[:len(response)-k] @ response[k:] for k in range(length)])

    def test_exact_stationary_likelihood_whitening_and_forecasts(self):
        for ar, ma in (([.7], []), ([], [-.5, .2]), ([.4, -.1], [-.3]),
                       ([.3], [-.3]), ([], []), ([.3], [.2, -.1, .05])):
            with self.subTest(ar=ar, ma=ma):
                data = self.fixture.data(n=2, t=25)
                data = data[~((data.id == 0) & (data.t >= 20))]
                panel = self.fixture.panel(order=(len(ar), len(ma), 0), data=data)
                args = self.fixture.args(panel)
                args[list(panel.args.positions['rho'])] = ar
                args[list(panel.args.positions['lambda'])] = ma
                ll = LL(args, panel)
                self.assertIsNotNone(ll.LL, ll.err_msg)
                self.assertIsInstance(ll.covariance, LowRankGaussianCovariance)
                expected_ll = 0.
                for group, times, block in ll.covariance.blocks:
                    u = ll.u[group, times, 0]
                    S = .8*self.reference(ar, ma, len(u)+1)
                    L = cholesky(S[:-1, :-1], lower=True)
                    w = solve_triangular(L, u, lower=True)
                    expected_ll -= .5*(len(u)*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+w@w)
                    np.testing.assert_allclose(block.density,
                        -.5*np.log(2*np.pi)-np.log(np.diag(L))-.5*w*w, atol=1e-11)
                    B = np.eye(len(u))
                    b = ll.AMA_1AR[0]
                    for lag in range(1, len(u)):
                        B[np.arange(lag, len(u)), np.arange(len(u)-lag)] = b[lag]
                    C = B @ S[:-1, :-1] @ B.T
                    np.testing.assert_allclose(np.eye(len(u))+block.H@block.P@block.H.T, C/.8, atol=1e-11)
                    np.testing.assert_allclose(block.whiten(block.e[:, None])[:, 0], w, atol=1e-11)
                    self.assertAlmostEqual(block.loglike, block.density.sum(), places=10)
                    mean = S[-1, :-1]@np.linalg.solve(S[:-1, :-1], u)
                    variance = S[-1, -1]-S[-1, :-1]@np.linalg.solve(S[:-1, :-1], S[:-1, -1])
                    correction, adjustment_var = ll.covariance.forecast_adjustment()
                    conditional = (np.asarray(ar) @ u[-len(ar):][::-1] if ar else 0.)
                    if ma:
                        conditional += np.asarray(ma) @ block.e[-len(ma):][::-1]
                    self.assertAlmostEqual(conditional+correction[group, 0], mean, places=10)
                    self.assertAlmostEqual(.8+adjustment_var[group, 0], variance, places=10)
                self.assertAlmostEqual(ll.LL, expected_ll, places=10)

    def test_full_analytic_derivatives_and_seasonal_cross_blocks(self):
        for order, seasonal in (((2, 1, 0), False), ((1, 3, 0), False),
                                ((0, 0, 0), False), ((0, 13, 0), True)):
            with self.subTest(order=order):
                panel = self.fixture.panel(order=order, data=self.fixture.data(n=1, t=35))
                args = self.fixture.args(panel)
                positions = list(panel.args.positions['lambda'])
                if seasonal:
                    args[positions] = 0
                    args[positions[0]], args[positions[11]], args[positions[12]] = -.4, -.55, .22
                ll = LL(args, panel)
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
                    np.testing.assert_allclose(scores[:, :, i], (a.ll_value-b.ll_value)[:, :, 0]/(2*step),
                                               atol=2e-5, rtol=2e-5)
                    numeric = (gradient(panel).get(a)[0]-gradient(panel).get(b)[0])/(2*step)
                    np.testing.assert_allclose(H[:, i], numeric, atol=3e-4, rtol=2e-5)

    def test_near_boundary_prior_and_cancellation(self):
        panel = self.fixture.panel(data=self.fixture.data(n=1, t=30))
        for ar, ma in ((.999, -.2), (.4, -.4), (.2, -.999)):
            args = self.fixture.args(panel)
            args[list(panel.args.positions['rho'])] = ar
            args[list(panel.args.positions['lambda'])] = ma
            ll = LL(args, panel)
            self.assertIsNotNone(ll.LL, ll.err_msg)
            block = ll.covariance.blocks[0][2]
            self.assertAlmostEqual(block.P[0, 0], 1/(1-ar*ar), places=8)
            if ar == .999:
                # This must include the infinite tail, not 30 presample shocks.
                expected = (.8*(1+ma*ma+2*ar*ma)/(1-ar*ar))
                implied = .8*(1+block.H[0] @ block.P @ block.H[0])
                self.assertAlmostEqual(implied, expected, places=8)
            g = gradient(panel)
            score, scores = g.get(ll)
            H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
            self.assertTrue(np.isfinite(H).all())
            for i in range(len(args)):
                step = 1e-7
                up, down = args.copy(), args.copy()
                up[i] += step
                down[i] -= step
                a, b = LL(up, panel), LL(down, panel)
                self.assertIsNotNone(a.LL, a.err_msg)
                self.assertIsNotNone(b.LL, b.err_msg)
                np.testing.assert_allclose(score[i], (a.LL-b.LL)/(2*step),
                                           atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(scores[:, :, i],
                    (a.ll_value-b.ll_value)[:, :, 0]/(2*step), atol=2e-5, rtol=2e-5)
                numeric = (gradient(panel).get(a)[0]-gradient(panel).get(b)[0])/(2*step)
                np.testing.assert_allclose(H[:, i], numeric, atol=1e-3, rtol=2e-5)

    def test_long_history_never_builds_dense_covariance(self):
        panel = self.fixture.panel(order=(2, 2, 0), data=self.fixture.data(n=1, t=4000))
        with patch.object(CovarianceState, '__init__', side_effect=AssertionError("Dense covariance used")):
            ll = LL(self.fixture.args(panel), panel)
            self.assertIsNotNone(ll.LL, ll.err_msg)
            g = gradient(panel)
            g.get(ll)
            H = hessian(panel, g).get(ll, *ll.llfunc.hessian())
        self.assertTrue(np.isfinite(H).all())
        block = ll.covariance.blocks[0][2]
        self.assertEqual(block.H.shape, (4000, 2))
        self.assertEqual(block.factor.shape, (2, 2))
        self.assertEqual(block.state.ddH.shape, (4, 4, 4001, 2))

    def test_singular_observable_covariance_is_supported(self):
        # Correct inverse MA column; common AR/MA factors cancel observations,
        # not the positive stationary state prior.
        state = StationaryLoading((.3**np.arange(20)), [.3], [-.3], 20)
        np.testing.assert_allclose(state.H, 0., atol=1e-15)

    def test_orders_longer_than_observed_history(self):
        ar, ma, length = [.4, -.1], [-.3, .15, .1, -.05, .02], 3
        impulse = np.eye(1, length)[0]
        inverse_ma = lfilter([1.], np.r_[1., ma], impulse)
        B = toeplitz(lfilter(np.r_[1., -np.asarray(ar)], np.r_[1., ma], impulse),
                     impulse)
        state = StationaryLoading(inverse_ma, ar, ma, length)
        block = LowRankBlock(state, np.array([.2, -.3, .1]), .8)
        expected = B @ self.reference(ar, ma, length) @ B.T
        np.testing.assert_allclose(np.eye(length)+block.H@block.P@block.H.T,
                                   expected, atol=1e-12)
        self.assertAlmostEqual(block.loglike, block.density.sum(), places=12)
        state.derivatives(2)
        self.assertTrue(np.isfinite(state.ddH).all())

    def test_hessian_batches_small_system_solves(self):
        panel = self.fixture.panel(order=(0, 13, 0), data=self.fixture.data(n=1, t=35))
        ll = LL(self.fixture.args(panel), panel)
        g = gradient(panel)
        g.get(ll)
        block = ll.covariance.blocks[0][2]
        E, V = ll.covariance._directions(g)
        with patch.object(low_rank, 'cho_solve', wraps=low_rank.cho_solve) as solve:
            with patch.object(low_rank, 'solve_triangular',
                              wraps=low_rank.solve_triangular) as triangular:
                first = block.curvature(E[0], V[0, 0], ll.covariance.positions)
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(triangular.call_count, 2)
        self.assertTrue(all(call.args[1].ndim == 2 for call in solve.call_args_list))
        changed = E[0].copy()
        changed[:, 0] += .2
        second = block.curvature(changed, V[0, 0], ll.covariance.positions)
        self.assertFalse(np.allclose(first, second))
        np.testing.assert_array_equal(
            block.curvature(E[0], V[0, 0], ll.covariance.positions), first)
