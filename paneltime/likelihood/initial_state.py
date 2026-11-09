"""Stationary Gaussian ARMA initial-state integration and analytic derivatives.

The zero-history innovation vector is e = B u. A stationary presample state
adds covariance v0 B O P O' B', where P solves a unit-noise Lyapunov equation.
The existing conditional variance path D and effects-adjusted e are retained:
C = D + v0 B O P O' B'. This is exact for homoskedastic Gaussian ARMA without
effects; with GARCH/effects it is a conditional initial-state approximation.
"""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_discrete_lyapunov, solve_triangular


class StationaryState:
    """Covariance of the initial-state contribution to zero-history innovations."""

    def __init__(self, ar, ma, length):
        self.ar = np.asarray(ar, dtype=float).reshape(-1)
        self.ma = np.asarray(ma, dtype=float).reshape(-1)
        self.length = length
        self.p, self.q = len(self.ar), len(self.ma)
        self.size = max(self.p, self.q)
        self.count = self.p + self.q
        r = self.size
        self.A = np.zeros((r, r))
        if not r:
            self.P = self.A.copy()
            self.O = self.H = np.zeros((length, 0))
            self.W = np.zeros((length, length))
            self._order = 0
            return
        self.A[0, :self.p] = self.ar
        self.A[1:, :-1] = np.eye(r - 1)
        if np.max(np.abs(np.linalg.eigvals(self.A))) >= 1:
            raise np.linalg.LinAlgError("Stationary ARIMA initialization requires stationary AR roots.")
        z = np.zeros(r)
        z[:self.p] += self.ar
        z[:self.q] += self.ma
        noise = np.zeros((r, r))
        noise[0, 0] = 1
        self.P = solve_discrete_lyapunov(self.A, noise)
        self.O = np.empty((length, r))
        self.O[0] = z
        for t in range(1, length):
            self.O[t] = self.O[t - 1] @ self.A
        self.H = np.zeros_like(self.O)
        for t in range(length):
            self.H[t] = self.O[t]
            for j in range(min(t, self.p)):
                self.H[t] -= self.ar[j] * self.O[t - j - 1]
            for j in range(min(t, self.q)):
                self.H[t] -= self.ma[j] * self.H[t - j - 1]
        self.W = self.H @ self.P @ self.H.T
        self._order = 0

    def derivatives(self, order=1):
        if self._order >= order:
            return
        n, r, length = self.count, self.size, self.length
        if not r:
            self.dW = np.zeros((0, length, length))
            self.ddW = np.zeros((0, 0, length, length))
            self._order = order
            return
        A, P, O, H = self.A, self.P, self.O, self.H
        dA = np.zeros((n, r, r))
        dA[np.arange(self.p), 0, np.arange(self.p)] = 1
        dZ = np.zeros((n, r))
        dZ[np.arange(self.p), np.arange(self.p)] = 1
        dZ[self.p + np.arange(self.q), np.arange(self.q)] = 1
        dP = np.empty((n, r, r))
        for i in range(n):
            rhs = dA[i] @ P @ A.T + A @ P @ dA[i].T
            dP[i] = solve_discrete_lyapunov(A, rhs)
        dO = np.zeros((n, length, r))
        dH = np.zeros_like(dO)
        dO[:, 0] = dZ
        for t in range(1, length):
            dO[:, t] = dO[:, t - 1] @ A + O[t - 1] @ dA
        for t in range(length):
            dH[:, t] = dO[:, t]
            for j in range(min(t, self.p)):
                dH[:, t] -= self.ar[j] * dO[:, t - j - 1]
                dH[j, t] -= O[t - j - 1]
            for j in range(min(t, self.q)):
                dH[:, t] -= self.ma[j] * dH[:, t - j - 1]
                dH[self.p + j, t] -= H[t - j - 1]
        self.dW = (
            dH @ P @ H.T + H @ dP @ H.T
            + H @ P @ dH.swapaxes(-1, -2)
        )
        self._order = 1
        if order == 1:
            return
        ddP = np.empty((n, n, r, r))
        ddO = np.zeros((n, n, length, r))
        ddH = np.zeros_like(ddO)
        for i in range(n):
            for j in range(n):
                rhs = (
                    dA[i] @ dP[j] @ A.T + dA[i] @ P @ dA[j].T
                    + dA[j] @ dP[i] @ A.T + A @ dP[i] @ dA[j].T
                    + dA[j] @ P @ dA[i].T + A @ dP[j] @ dA[i].T
                )
                ddP[i, j] = solve_discrete_lyapunov(A, rhs)
                for t in range(1, length):
                    ddO[i, j, t] = (
                        ddO[i, j, t - 1] @ A
                        + dO[i, t - 1] @ dA[j] + dO[j, t - 1] @ dA[i]
                    )
                for t in range(length):
                    ddH[i, j, t] = ddO[i, j, t]
                    for lag in range(min(t, self.p)):
                        previous = t - lag - 1
                        ddH[i, j, t] -= self.ar[lag] * ddO[i, j, previous]
                        if i == lag:
                            ddH[i, j, t] -= dO[j, previous]
                        if j == lag:
                            ddH[i, j, t] -= dO[i, previous]
                    for lag in range(min(t, self.q)):
                        previous = t - lag - 1
                        ddH[i, j, t] -= self.ma[lag] * ddH[i, j, previous]
                        if i == self.p + lag:
                            ddH[i, j, t] -= dH[j, previous]
                        if j == self.p + lag:
                            ddH[i, j, t] -= dH[i, previous]
        self.ddW = np.empty((n, n, length, length))
        for i in range(n):
            for j in range(n):
                self.ddW[i, j] = (
                    ddH[i, j] @ P @ H.T + dH[i] @ dP[j] @ H.T
                    + dH[i] @ P @ dH[j].T + dH[j] @ dP[i] @ H.T
                    + H @ ddP[i, j] @ H.T + H @ dP[i] @ dH[j].T
                    + dH[j] @ P @ dH[i].T + H @ dP[j] @ dH[i].T
                    + H @ P @ ddH[i, j].T
                )
        self._order = 2


class InitialStateLikelihood:
    """Gaussian block densities, pointwise scores and Hessian curvature."""

    def __init__(self, panel, ll):
        self.blocks = []
        self.shape = ll.e_RE.shape
        self.positions = list(panel.args.positions['rho']) + list(panel.args.positions['lambda'])
        self.count = panel.args.n_args
        self.ll_value = np.zeros(self.shape)
        self.partial_e = np.zeros(self.shape)
        self.partial_v = np.zeros(self.shape)
        signature = tuple(ll.args.args_d['rho'].ravel()), tuple(ll.args.args_d['lambda'].ravel())
        cached_signature, cache = getattr(panel, '_initial_state_cache', (None, {}))
        if cached_signature != signature:
            cache = {}
            panel._initial_state_cache = signature, cache
        d = panel.pqdkm[2]
        for group in range(panel.N):
            times = np.flatnonzero(panel.included[3][group, :, 0])
            times = times[times >= d]
            if not len(times):
                continue
            length = len(times)
            if length not in cache:
                cache[length] = StationaryState(
                    ll.args.args_d['rho'], ll.args.args_d['lambda'], length,
                )
            state = cache[length]
            e = ll.e_RE[group, times, 0]
            v = ll.llfunc.var[group, times, 0]
            if not np.all(np.isfinite(v)) or np.any(v <= 0) or not np.array_equal(v, ll.var[group, times, 0]):
                raise np.linalg.LinAlgError("Stationary ARIMA initialization requires positive, unclipped conditional variances.")
            C = np.diag(v) + v[0] * state.W
            L = cholesky(C, lower=True)
            standardized = solve_triangular(L, e, lower=True)
            inverse = cho_solve((L, True), np.eye(length))
            alpha = inverse @ e
            weight = 0.5 * (np.outer(alpha, alpha) - inverse)
            self.ll_value[group, times, 0] = (
                -0.5 * np.log(2 * np.pi) - np.log(np.diag(L)) - 0.5 * standardized**2
            )
            self.partial_e[group, times, 0] = -alpha
            self.partial_v[group, times, 0] = np.diag(weight)
            self.partial_v[group, times[0], 0] += np.sum(weight * state.W)
            self.blocks.append((group, times, state, e, v, L, inverse, alpha, weight))

    def whiten(self, values):
        result = np.zeros_like(values)
        for group, times, state, e, v, L, inverse, alpha, weight in self.blocks:
            result[group, times] = solve_triangular(L, values[group, times], lower=True)
        return result

    def forecast_adjustment(self):
        mean = np.zeros((self.shape[0], 1))
        variance = np.zeros_like(mean)
        for group, times, state, e, v, L, inverse, alpha, weight in self.blocks:
            extended = StationaryState(state.ar, state.ma, len(times) + 1)
            cross = v[0] * extended.W[-1, :-1]
            mean[group, 0] = cross @ alpha
            variance[group, 0] = v[0] * extended.W[-1, -1] - cross @ inverse @ cross
        return mean, variance

    def _directions(self, g):
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
        dc = np.zeros((self.count, len(v), len(v)))
        for i in range(self.count):
            dc[i] = np.diag(dv[:, i]) + dv[0, i] * state.W
        dc[self.positions] += v[0] * state.dW
        return dc

    def scores(self, g):
        de, dv = self._directions(g)
        scores = np.zeros_like(de)
        for group, times, state, e, v, L, inverse, alpha, weight in self.blocks:
            dc = self._covariance_directions(state, v, dv[group, times])
            w = solve_triangular(L, e, lower=True)
            for i in range(self.count):
                scaled = solve_triangular(L, dc[i], lower=True)
                scaled = solve_triangular(L, scaled.T, lower=True).T
                phi = np.tril(scaled)
                phi[np.diag_indices_from(phi)] *= 0.5
                dw = solve_triangular(L, de[group, times, i], lower=True) - phi @ w
                scores[group, times, i] = -np.diag(phi) - w * dw
        return scores.sum(axis=(0, 1)), scores

    def hessian_correction(self, g, local_curvature):
        """Pullback curvature excluding e'' and v'', handled by calculus.py."""
        de, dv = self._directions(g)
        result = np.zeros((self.count, self.count))
        ee, ve, vv = local_curvature
        for group, times, state, e, v, L, inverse, alpha, weight in self.blocks:
            state.derivatives(2)
            E, V = de[group, times], dv[group, times]
            dc = self._covariance_directions(state, v, V)
            da = np.stack([inverse @ (E[:, i] - dc[i] @ alpha)
                           for i in range(self.count)])
            inverse_dc = inverse @ dc
            dw = 0.5 * (
                da[:, :, None] * alpha[None, None, :]
                + alpha[None, :, None] * da[:, None, :]
                + inverse_dc @ inverse
            )
            shape_index = {position: index for index, position in enumerate(self.positions)}
            for i in range(self.count):
                for j in range(i + 1):
                    ddc = np.zeros_like(state.W)
                    if i in shape_index:
                        ddc += V[0, j] * state.dW[shape_index[i]]
                    if j in shape_index:
                        ddc += V[0, i] * state.dW[shape_index[j]]
                    if i in shape_index and j in shape_index:
                        ddc += v[0] * state.ddW[shape_index[i], shape_index[j]]
                    value = -E[:, i] @ da[j] + np.sum(dw[j] * dc[i]) + np.sum(weight * ddc)
                    # The existing Hessian already includes the pointwise normal
                    # curvature. Replace it, preserving its weighted e''/v'' terms.
                    value -= np.sum(
                        ee[group, times, 0] * E[:, i] * E[:, j]
                        + ve[group, times, 0] * (V[:, i] * E[:, j] + E[:, i] * V[:, j])
                        + vv[group, times, 0] * V[:, i] * V[:, j]
                    )
                    result[i, j] += value
                    if i != j:
                        result[j, i] += value
        return result
