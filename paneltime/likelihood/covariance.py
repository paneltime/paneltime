"""Dispatch and calculus integration for optional Gaussian covariance corrections."""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, toeplitz
from scipy.signal import lfilter
from .low_rank import StationaryLoading, LowRankBlock


def lag_matrix(column):
    row = np.zeros_like(column)
    row[0] = column[0]
    return toeplitz(column, row)


def convolve(a, b):
    return np.convolve(a, b)[:len(a)]


class CovarianceState:
    """Finite impulse history for the dense GARCH/effects approximation."""

    def __init__(self, inverse, ar, ma):
        self.b = np.asarray(inverse[0], dtype=float)
        self.ar = np.asarray(ar).ravel()
        self.ma = np.asarray(ma).ravel()
        if len(self.ar) and np.any(np.abs(np.roots(np.r_[1., -self.ar])) >= 1):
            raise np.linalg.LinAlgError("Covariance likelihood requires stationary AR roots.")
        self.length = len(self.b)
        self.count = len(self.ar) + len(self.ma)
        unit = np.zeros(self.length)
        unit[0] = 1
        self.psi = lfilter([1.], self.b, unit)
        self.autocovariances = np.correlate(self.psi, self.psi, mode='full')[self.length-1:]
        indices = np.arange(self.length)[:, None] + np.arange(self.length - 1) + 1
        self.indices = indices
        self.J = self._presample(self.psi)
        self.B = lag_matrix(self.b)
        self.H = self.B @ self.J
        self.W = self.H @ self.H.T
        self._order = 0

    def _presample(self, values):
        return np.r_[values, np.zeros(self.length)][self.indices]

    def derivatives(self, order=1):
        if self._order >= order:
            return
        n, T, p = self.count, self.length, len(self.ar)
        db = np.zeros((n, T))
        delta_ma = np.zeros((n, T))
        for i in range(n):
            lag = i + 1 if i < p else i - p + 1
            rhs = np.zeros(T)
            if lag < T:
                if i < p:
                    rhs[lag] = -1
                else:
                    delta_ma[i, lag] = 1
                    rhs[lag:] = -self.b[:-lag]
            db[i] = lfilter([1.], np.r_[1., self.ma], rhs)
        dpsi = np.array([-lfilter([1.], self.b, convolve(row, self.psi)) for row in db])
        dB = np.array([lag_matrix(row) for row in db]).reshape(n, T, T)
        dJ = np.array([self._presample(row) for row in dpsi]).reshape(n, T, T-1)
        dH = dB @ self.J + self.B @ dJ
        self.dW = dH @ self.H.T + self.H @ dH.swapaxes(-1, -2)
        self._order = 1
        if order == 1:
            return
        self.ddW = np.zeros((n, n, T, T))
        for i in range(n):
            for j in range(i + 1):
                ddb = -lfilter([1.], np.r_[1., self.ma],
                               convolve(delta_ma[i], db[j]) + convolve(delta_ma[j], db[i]))
                ddpsi = -lfilter([1.], self.b,
                    convolve(ddb, self.psi) + convolve(db[i], dpsi[j]) + convolve(db[j], dpsi[i]))
                ddH = (lag_matrix(ddb) @ self.J + dB[i] @ dJ[j] + dB[j] @ dJ[i]
                       + self.B @ self._presample(ddpsi))
                value = (ddH @ self.H.T + self.H @ ddH.T
                         + dH[i] @ dH[j].T + dH[j] @ dH[i].T)
                self.ddW[i, j] = self.ddW[j, i] = value
        self._order = 2


class GaussianCovariance:
    """Dense finite-history density, with its correction to conditional density."""

    def __init__(self, panel, ll):
        self._initialize_results(panel, ll)
        self.covariance_lags = np.zeros(self.shape[:2])
        inverse = ll.AMA_1AR
        signature = (tuple(inverse[0]), tuple(ll.args.args_d['rho'].ravel()),
                     tuple(ll.args.args_d['lambda'].ravel()), panel.pqdkm[2])
        states = self._cached_states(panel, '_covariance_cache', signature)
        for group, times in self._observed_groups(panel):
            length = len(times)
            if length not in states:
                states[length] = CovarianceState((inverse[0][:length], inverse[1]),
                    ll.args.args_d['rho'], ll.args.args_d['lambda'])
            self._add_dense_block(group, times, states[length], ll)
        self.correction = self.ll_value - ll.llfunc.ll_value

    def _initialize_results(self, panel, ll):
        self.shape = ll.e_RE.shape
        self.count = panel.args.n_args
        self.positions = list(panel.args.positions['rho']) + list(panel.args.positions['lambda'])
        self.ll_value = np.zeros(self.shape)
        self.partial_e = np.zeros(self.shape)
        self.partial_v = np.zeros(self.shape)
        self.blocks = []
        self.group_ll = np.zeros(panel.N)
        self.group_correction = np.zeros(panel.N)

    @staticmethod
    def _cached_states(panel, cache_name, signature):
        """Share loadings by group length, retaining only the current coefficients."""
        cached_signature, states = getattr(panel, cache_name, (None, {}))
        if signature != cached_signature:
            states = {}
            setattr(panel, cache_name, (signature, states))
        return states

    @staticmethod
    def _observed_groups(panel):
        for group in range(panel.N):
            times = np.flatnonzero(panel.included[3][group, :, 0])
            times = times[times >= panel.pqdkm[2]]
            if len(times):
                yield group, times

    @staticmethod
    def _variance(ll, group, times, constant=False):
        v = ll.llfunc.var[group, times, 0]
        if (np.any(v <= 0) or not np.isfinite(v).all()
                or not np.array_equal(v, ll.var[group, times, 0])):
            raise np.linalg.LinAlgError("Covariance likelihood requires positive unclipped variances.")
        if constant and not np.all(v == v[0]):
            raise np.linalg.LinAlgError("Low-rank ARMA likelihood requires constant variance.")
        return v

    def _record_density(self, group, times, density, loglike, ll):
        self.ll_value[group, times, 0] = density
        self.group_ll[group] = loglike
        self.group_correction[group] = loglike - ll.llfunc.ll_value[group].sum()

    def _add_dense_block(self, group, times, state, ll):
        e = ll.e_RE[group, times, 0]
        v = self._variance(ll, group, times)
        self.covariance_lags[group, :len(times)] = v[0] * state.autocovariances
        C = np.diag(v) + v[0] * state.W
        L = cholesky(C, lower=True)
        w = solve_triangular(L, e, lower=True)
        K = cho_solve((L, True), np.eye(len(times)))
        alpha = K @ e
        weight = .5 * (np.outer(alpha, alpha) - K)
        density = -.5*np.log(2*np.pi) - np.log(np.diag(L)) - .5*w*w
        self._record_density(group, times, density, density.sum(), ll)
        self.partial_e[group, times, 0] = -alpha
        self.partial_v[group, times, 0] = np.diag(weight)
        self.partial_v[group, times[0], 0] += np.sum(weight * state.W)
        self.blocks.append((group, times, state, e, v, L, K, alpha, weight))

    def whiten(self, values):
        result = np.zeros_like(values)
        for group, times, _state, _e, _v, L, _K, _alpha, _weight in self.blocks:
            result[group, times] = solve_triangular(L, values[group, times], lower=True)
        return result

    def forecast_adjustment(self):
        mean = np.zeros((self.shape[0], 1))
        variance = np.zeros_like(mean)
        for group, _times, state, e, v, _L, K, alpha, _weight in self.blocks:
            # The truncated impulse sequence defines a finite MA process.
            bnext = -state.psi[1:] @ state.b[:0:-1]
            actual_next = -sum(state.ma[j]*state.b[-j-1]
                               for j in range(min(len(state.ma), len(state.b))))
            if len(state.b) <= len(state.ar):
                actual_next -= state.ar[len(state.b)-1]
            row = np.zeros(len(state.psi)-1)
            for lag in range(1, len(state.b)):
                row += state.b[lag] * state.J[-lag]
            row += bnext * state.J[0]
            cross = v[0] * row @ state.H.T
            mean[group, 0] = cross @ alpha + (actual_next-bnext)*e[0]
            variance[group, 0] = v[0] * (row @ row) - cross @ K @ cross
        return mean, variance

    def _directions(self, g):
        """Collect existing regression/ARMA/effects derivatives in full coordinates."""
        de = np.zeros(self.shape[:2] + (self.count,))
        dv = np.zeros_like(de)
        for name, positions in g.panel.args.positions.items():
            if not len(positions):
                continue
            if name in ('beta', 'rho', 'lambda'):
                residual = getattr(g, f'de_{name}_RE')
                variance = getattr(g, f'dvar_sigma_{name}')
                if residual is not None:
                    de[:, :, list(positions)] = residual
            else:
                variance = getattr(g, f'dvar_{name}', None)
            if variance is not None:
                dv[:, :, list(positions)] = variance
        return de, dv

    def _covariance_directions(self, state, v, dv):
        state.derivatives()
        dc = np.array([np.diag(dv[:, i]) + dv[0, i]*state.W for i in range(self.count)])
        dc[self.positions] += v[0]*state.dW
        return dc

    def scores(self, g):
        de, dv = self._directions(g)
        scores = np.zeros_like(de)
        for group, times, state, e, v, L, _K, _alpha, _weight in self.blocks:
            dc = self._covariance_directions(state, v, dv[group, times])
            w = solve_triangular(L, e, lower=True)
            for i in range(self.count):
                scaled = solve_triangular(L, dc[i], lower=True)
                scaled = solve_triangular(L, scaled.T, lower=True).T
                phi = np.tril(scaled)
                phi[np.diag_indices_from(phi)] *= .5
                dw = solve_triangular(L, de[group, times, i], lower=True) - phi @ w
                scores[group, times, i] = -np.diag(phi) - w*dw
        return scores.sum(axis=(0, 1)), scores

    def hessian_correction(self, g, local_curvature):
        de, dv = self._directions(g)
        result = np.zeros((self.count, self.count))
        ee, ve, vv = local_curvature
        shape_index = {position: index for index, position in enumerate(self.positions)}
        for group, times, state, _e, v, _L, K, alpha, weight in self.blocks:
            state.derivatives(2)
            E, V = de[group, times], dv[group, times]
            dc = self._covariance_directions(state, v, V)
            da = np.array([K @ (E[:, i] - dc[i] @ alpha) for i in range(self.count)])
            dw = .5*(da[:, :, None]*alpha[None, None, :]
                      + alpha[None, :, None]*da[:, None, :] + K @ dc @ K)
            for i in range(self.count):
                for j in range(i + 1):
                    ddc = np.zeros_like(state.W)
                    if i in shape_index:
                        ddc += V[0, j]*state.dW[shape_index[i]]
                    if j in shape_index:
                        ddc += V[0, i]*state.dW[shape_index[j]]
                    if i in shape_index and j in shape_index:
                        ddc += v[0]*state.ddW[shape_index[i], shape_index[j]]
                    value = -E[:, i] @ da[j] + np.sum(dw[j]*dc[i]) + np.sum(weight*ddc)
                    value -= np.sum(
                        ee[group, times, 0]*E[:, i]*E[:, j]
                        + ve[group, times, 0]*(V[:, i]*E[:, j] + E[:, i]*V[:, j])
                        + vv[group, times, 0]*V[:, i]*V[:, j])
                    result[i, j] += value
                    if i != j:
                        result[j, i] += value
        return result


class LowRankGaussianCovariance(GaussianCovariance):
    """Exact stationary ARMA path; the existing general approximation is separate."""

    def __init__(self, panel, ll):
        self._initialize_results(panel, ll)
        ar, ma = ll.args.args_d['rho'], ll.args.args_d['lambda']
        signature = tuple(ar.ravel()), tuple(ma.ravel()), tuple(ll.AMA_1[0])
        states = self._cached_states(panel, '_low_rank_cache', signature)
        for group, times in self._observed_groups(panel):
            length = len(times)
            if length not in states:
                states[length] = StationaryLoading(ll.AMA_1[0][:length], ar, ma, length)
            v = self._variance(ll, group, times, constant=True)
            block = LowRankBlock(states[length], ll.e_RE[group, times, 0], v[0])
            self._record_density(group, times, block.density, block.loglike, ll)
            self.partial_e[group, times, 0] = block.partial_e
            self.partial_v[group, times, 0] = block.partial_v
            self.blocks.append((group, times, block))
        self.correction = self.ll_value-ll.llfunc.ll_value

    def whiten(self, values):
        result = np.zeros_like(values)
        for group, times, block in self.blocks:
            result[group, times] = block.whiten(values[group, times])
        return result

    def forecast_adjustment(self):
        mean = np.zeros((self.shape[0], 1))
        variance = np.zeros_like(mean)
        for group, _times, block in self.blocks:
            next_h = block.state.H[-1]
            mean[group, 0] = next_h @ block.x
            variance[group, 0] = block.v*(next_h @ block.R @ next_h)
        return mean, variance

    def scores(self, g):
        E, V = self._directions(g)
        scores = np.zeros_like(E)
        for group, times, block in self.blocks:
            scores[group, times] = block.scores(E[group, times], V[group, times[0]], self.positions)
        return scores.sum(axis=(0, 1)), scores

    def hessian_correction(self, g, local_curvature):
        E, V = self._directions(g)
        ee, ve, vv = local_curvature
        result = np.zeros((self.count, self.count))
        for group, times, block in self.blocks:
            e, v = E[group, times], V[group, times]
            result += block.curvature(e, v[0], self.positions)
            result -= (e.T @ (ee[group, times]*e)+v.T @ (ve[group, times]*e)
                       + e.T @ (ve[group, times]*v)+v.T @ (vv[group, times]*v))
        return result


def covariance_likelihood(panel, ll):
    options = panel.options
    arma_only = (not any(panel.pqdkm[3:]) and panel.W.shape[2] == 1
                 and not options.fixed_random_group_eff and not options.fixed_random_time_eff
                 and not options.fixed_random_variance_eff)
    if arma_only:
        return LowRankGaussianCovariance(panel, ll)
    return GaussianCovariance(panel, ll)
