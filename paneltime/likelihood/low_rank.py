"""Stationary Gaussian ARMA likelihood using small presample covariance systems."""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_discrete_lyapunov, solve_triangular
from scipy.signal import lfilter


class StationaryLoading:
    """Stationary prior P and boundary loadings H, including one forecast row.

    A is the AR companion matrix, O the unfiltered state loading, and D its
    finite AR-filtered boundary. H = M^-1 D reuses the native inverse-MA column.
    Derivative axes follow the AR coefficients, then the MA coefficients.
    """

    def __init__(self, inverse_ma, ar, ma, length):
        self.ar, self.ma = np.asarray(ar).ravel(), np.asarray(ma).ravel()
        self.p, self.q = len(self.ar), len(self.ma)
        self.r, self.n, self.length = max(self.p, self.q), self.p+self.q, length
        self.A, self.P = self._stationary_prior()
        self.O, self.D = self._boundary_loadings()
        self.H = self._innovation_loadings(inverse_ma)
        self._order = 0

    def _stationary_prior(self):
        r = self.r
        A = np.zeros((r, r))
        if not r:
            return A, np.zeros((0, 0))
        A[0, :self.p] = self.ar
        A[1:, :-1] = np.eye(r-1)
        if np.max(np.abs(np.linalg.eigvals(A))) >= 1:
            raise np.linalg.LinAlgError("Covariance likelihood requires stationary AR roots.")
        noise = np.zeros((r, r))
        noise[0, 0] = 1
        return A, solve_discrete_lyapunov(A, noise)

    def _boundary_loadings(self):
        r = self.r
        z = np.zeros(r)
        z[:self.p] += self.ar
        z[:self.q] += self.ma
        O = np.zeros((r, r))
        if r:
            O[0] = z
        for t in range(1, r):
            O[t] = O[t-1] @ self.A
        D = O.copy()
        for t in range(r):
            for j in range(min(t, self.p)):
                D[t] -= self.ar[j]*O[t-j-1]
        return O, D

    def _innovation_loadings(self, inverse_ma):
        length, r = self.length, self.r
        H = np.zeros((length+1, r))
        inverse_ma = np.asarray(inverse_ma).ravel()
        next_coefficient = -sum(self.ma[j]*inverse_ma[-j-1]
                                for j in range(min(self.q, length)))
        column = np.r_[inverse_ma[:length], next_coefficient]
        for j in range(min(r, length+1)):
            H[j:] += column[:length+1-j, None]*self.D[j]
        return H

    def derivatives(self, order=1):
        """Populate analytic first/second derivatives, reusing this state's work."""
        if self._order >= order:
            return
        dA, dO = self._first_derivatives()
        self._order = 1
        if order == 2:
            self._second_derivatives(dA, dO)
            self._order = 2

    def _first_derivatives(self):
        n, r, p, q, T = self.n, self.r, self.p, self.q, self.length+1
        dA = np.zeros((n, r, r))
        if p:
            dA[np.arange(p), 0, np.arange(p)] = 1
        dO = np.zeros((n, r, r))
        if r:
            dO[np.arange(p), 0, np.arange(p)] = 1
            dO[p+np.arange(q), 0, np.arange(q)] += 1
        self.dP = np.zeros((n, r, r))
        for i in range(p):
            self.dP[i] = solve_discrete_lyapunov(self.A,
                dA[i] @ self.P @ self.A.T + self.A @ self.P @ dA[i].T)
        for t in range(1, r):
            dO[:, t] = dO[:, t-1] @ self.A + self.O[t-1] @ dA
        dD = dO.copy()
        for t in range(r):
            for j in range(min(t, p)):
                dD[:, t] -= self.ar[j]*dO[:, t-j-1]
                dD[j, t] -= self.O[t-j-1]
        self.dH = np.zeros((n, T, r))
        for i in range(n):
            rhs = np.zeros((T, r))
            rhs[:min(T, r)] = dD[i, :min(T, r)]
            if i >= p:
                lag = i-p+1
                rhs[lag:] -= self.H[:-lag]
            self.dH[i] = lfilter([1.], np.r_[1., self.ma], rhs, axis=0)
        return dA, dO

    def _second_derivatives(self, dA, dO):
        n, r, p, T = self.n, self.r, self.p, self.length+1
        self.ddP = np.zeros((n, n, r, r))
        self.ddH = np.zeros((n, n, T, r))
        for i in range(n):
            for j in range(i+1):
                if i < p and j < p:
                    rhs = (
                        dA[i] @ self.dP[j] @ self.A.T + dA[i] @ self.P @ dA[j].T
                        + dA[j] @ self.dP[i] @ self.A.T + self.A @ self.dP[i] @ dA[j].T
                        + dA[j] @ self.P @ dA[i].T + self.A @ self.dP[j] @ dA[i].T)
                    self.ddP[i, j] = self.ddP[j, i] = solve_discrete_lyapunov(self.A, rhs)
                rhs = self._second_loading_rhs(i, j, dA, dO)
                value = lfilter([1.], np.r_[1., self.ma], rhs, axis=0)
                self.ddH[i, j] = self.ddH[j, i] = value

    def _second_loading_rhs(self, i, j, dA, dO):
        r, p, T = self.r, self.p, self.length+1
        ddO = np.zeros((r, r))
        for t in range(1, r):
            ddO[t] = (ddO[t-1] @ self.A + dO[i, t-1] @ dA[j]
                      + dO[j, t-1] @ dA[i])
        rhs = np.zeros((T, r))
        for t in range(min(T, r)):
            rhs[t] = ddO[t]
            for lag in range(min(t, p)):
                previous = t-lag-1
                rhs[t] -= self.ar[lag]*ddO[previous]
                if i == lag:
                    rhs[t] -= dO[j, previous]
                if j == lag:
                    rhs[t] -= dO[i, previous]
        for a, b in ((i, j), (j, i)):
            if a >= p:
                lag = a-p+1
                rhs[lag:] -= self.dH[b, :-lag]
        return rhs


class LowRankBlock:
    """One entity's density for Cov(e) = v (I + H P H').

    R is the posterior unit state covariance and x its posterior mean.
    G = H'H and w = H'e keep all factorization work in state coordinates.
    """

    def __init__(self, state, e, variance):
        self.state, self.e, self.v = state, e, variance
        self.H, self.P = state.H[:-1], state.P
        self._factor_covariance()
        self._density_partials()
        self._innovations()

    def _factor_covariance(self):
        H, P, r = self.H, self.P, self.state.r
        e, variance = self.e, self.v
        self.G = H.T @ H
        self.w = H.T @ e
        prior_factor = cholesky(P, lower=True) if r else P
        self.prior_factor = prior_factor
        small = np.eye(r) + prior_factor.T @ self.G @ prior_factor
        self.factor = cholesky(small, lower=True)
        self.R = prior_factor @ cho_solve((self.factor, True), prior_factor.T)
        self.x = self.R @ self.w
        self.alpha_unit = e-H @ self.x
        # The positive sum-of-squares form avoids cancellation in e'e-w'Rw.
        self.Q = (self.alpha_unit @ self.alpha_unit
                  + np.linalg.norm(prior_factor.T @ (self.w-self.G @ self.x))**2)
        self.logdet = 2*np.log(np.diag(self.factor)).sum()
        self.loglike = -.5*(len(e)*np.log(2*np.pi*variance)+self.logdet+self.Q/variance)

    def _density_partials(self):
        """Local e/v partials used by calculus.py's residual/variance chain rule."""
        H, P, variance = self.H, self.P, self.v
        self.partial_e = -self.alpha_unit/variance
        diagK = 1-np.einsum('ti,ij,tj->t', H, self.R, H)
        self.partial_v = .5*((self.alpha_unit/variance)**2-diagK/variance)
        c = H.T @ self.alpha_unit
        # The first variance also scales the integrated presample covariance.
        self.partial_v[0] += .5*(c @ P @ c/variance**2
            - np.trace(self.G @ P-self.G @ self.R @ self.G @ P)/variance)

    def _innovations(self):
        """Sequential densities and gains for robust scores and whitening."""
        r = self.state.r
        S, a = self.P.copy(), np.zeros(r)
        self.gains = np.zeros_like(self.H)
        self.variances = np.zeros(len(self.e))
        self.innovations = np.zeros(len(self.e))
        for t, h in enumerate(self.H):
            sh = S @ h
            f = 1+h @ sh
            if not np.isfinite(f) or f <= 0:
                raise np.linalg.LinAlgError("Invalid low-rank innovations variance.")
            gain = sh/f
            innovation = self.e[t]-h @ a
            self.gains[t], self.variances[t], self.innovations[t] = gain, f, innovation
            a += gain*innovation
            S -= np.outer(sh, sh)/f
            S = .5*(S+S.T)
        self.density = -.5*(np.log(2*np.pi*self.v*self.variances)
                            + self.innovations**2/(self.v*self.variances))

    def whiten(self, values):
        a = np.zeros((self.state.r, values.shape[1]))
        result = np.empty_like(values)
        for t, h in enumerate(self.H):
            innovation = values[t]-h @ a
            result[t] = innovation/np.sqrt(self.v*self.variances[t])
            a += self.gains[t, :, None]*innovation
        return result

    def directions(self, count, positions, order):
        """Embed AR/MA loading derivatives in the full model parameter axes."""
        self.state.derivatives(order)
        dH = np.zeros((count, *self.H.shape))
        dP = np.zeros((count, *self.P.shape))
        dH[positions] = self.state.dH[:, :-1]
        dP[positions] = self.state.dP
        return dH, dP

    def scores(self, E, variance_directions, positions):
        """Observation scores; E[t, i] is the innovation derivative for parameter i."""
        n, r = E.shape[1], self.state.r
        dH, dP = self.directions(n, positions, 1)
        S, dS = self.P.copy(), dP.copy()
        a, da = np.zeros(r), np.zeros((n, r))
        scores = np.zeros_like(E)
        for t, h in enumerate(self.H):
            dh = dH[:, t]
            sh = S @ h
            dsh = dS @ h + dh @ S
            f = self.variances[t]
            df = dh @ sh + dsh @ h
            nu = self.innovations[t]
            dnu = E[t]-dh @ a-da @ h
            full_df = self.v*df+variance_directions*f
            full_f = self.v*f
            scores[t] = -.5*(full_df/full_f+2*nu*dnu/full_f-nu*nu*full_df/full_f**2)
            gain = self.gains[t]
            dgain = dsh/f-sh[None, :]*df[:, None]/f**2
            da += dgain*nu+gain[None, :]*dnu[:, None]
            a += gain*nu
            dS -= ((dsh[:, :, None]*sh[None, None, :]+sh[None, :, None]*dsh[:, None, :])/f
                   - np.outer(sh, sh)[None]*df[:, None, None]/f**2)
            S -= np.outer(sh, sh)/f
        return scores

    def curvature(self, E, variance_directions, positions):
        """Density curvature with affine e and variance; calculus adds e''/v''."""
        n = E.shape[1]
        rows, columns = np.tril_indices(n)
        dq, ddq, ddlogdet = self._quadratic_curvature(E, positions, rows, columns)
        vi, vj, v = variance_directions[rows], variance_directions[columns], self.v
        values = -.5*(-len(self.e)*vi*vj/v**2+ddlogdet+ddq/v
                       -(dq[rows]*vj+dq[columns]*vi)/v**2+2*self.Q*vi*vj/v**3)
        result = np.zeros((n, n))
        result[rows, columns] = result[columns, rows] = values
        return result

    def _solve_batched(self, matrices, vectors):
        """Solve B X = RHS for B = I + P G via its symmetric factorization."""
        r, count = self.state.r, len(vectors)
        rhs = np.concatenate((matrices.transpose(1, 0, 2).reshape(r, count*r),
                              vectors.T), axis=1)
        if r:
            rhs = self.prior_factor @ cho_solve((self.factor, True),
                solve_triangular(self.prior_factor, rhs, lower=True))
        return (rhs[:, :count*r].reshape(r, count, r).transpose(1, 0, 2),
                rhs[:, count*r:].T)

    def _quadratic_curvature(self, E, positions, rows, columns):
        """Differentiate Q and log|B|, batching each derivative order's solves."""
        H, P, G, w, x = self.H, self.P, self.G, self.w, self.x
        r = self.state.r
        dH, dP = self.directions(E.shape[1], positions, 2)
        dG = dH.swapaxes(1, 2) @ H + H.T @ dH
        dw = (dH.swapaxes(1, 2) @ self.e) + (H.T @ E).T
        dB = dP @ G+P @ dG
        solved_dB, dx = self._solve_batched(dB, dP @ w+dw @ P.T-dB @ x)
        dq = 2*E.T @ self.e-dw @ x-dx @ w
        index = {position: i for i, position in enumerate(positions)}
        pairs = len(rows)
        ddB = np.zeros((pairs, r, r))
        ddx_rhs = np.zeros((pairs, r))
        ddq = np.zeros(pairs)
        for pair, (i, j) in enumerate(zip(rows, columns)):
            ddH, ddP = np.zeros_like(H), np.zeros_like(P)
            if i in index and j in index:
                ddH = self.state.ddH[index[i], index[j], :-1]
                ddP = self.state.ddP[index[i], index[j]]
            ddG = (ddH.T @ H+H.T @ ddH+dH[i].T @ dH[j]+dH[j].T @ dH[i])
            ddw = ddH.T @ self.e+dH[i].T @ E[:, j]+dH[j].T @ E[:, i]
            ddB[pair] = ddP @ G+dP[i] @ dG[j]+dP[j] @ dG[i]+P @ ddG
            ddx_rhs[pair] = (ddP @ w+dP[i] @ dw[j]+dP[j] @ dw[i]+P @ ddw
                            -ddB[pair] @ x-dB[i] @ dx[j]-dB[j] @ dx[i])
            ddq[pair] = (2*E[:, i] @ E[:, j]-ddw @ x-dw[i] @ dx[j]
                        -dw[j] @ dx[i])
        solved_ddB, ddx = self._solve_batched(ddB, ddx_rhs)
        ddq -= ddx @ w
        ddlogdet = (np.trace(solved_ddB, axis1=1, axis2=2)
                    - np.einsum('kij,kji->k', solved_dB[columns], solved_dB[rows]))
        return dq, ddq, ddlogdet
