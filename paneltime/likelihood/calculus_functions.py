#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second-derivative helper functions for ARIMA/GARCH panel likelihoods.

The functions in this module intentionally keep the original public API.  Most
arrays follow one of these conventions:

* first derivatives: ``(N, T, k)``
* pairwise second derivatives before summing: ``(N, T, k, m)``
* summed Hessian blocks: ``(k, m)``

``None`` is used throughout to mean "this parameter block is absent in the
current model specification".  The small helper functions below therefore have
explicit ``ignore_none`` behaviour rather than relying on scattered ``if``
statements in the Hessian code.
"""

import numpy as np
from .. import functions as fu

EXTREME_VALUE = 1e100


def _clip_extreme(x):
    """Clip very large finite values while preserving ``None``."""
    if x is None:
        return None
    return np.clip(x, -EXTREME_VALUE, EXTREME_VALUE)


def _grad(g, name):
    """Return ``g.de_<name>_RE``."""
    return getattr(g, f"de_{name}_RE")


def _dvRE(g, name):
    """Return ``g.dvRE_<name>``."""
    return getattr(g, f"dvRE_{name}")


def dd_func_lags_mult(
    panel,
    ll,
    g,
    AMAL,
    vname1,
    vname2,
    transpose=False,
    u_gradient=False,
):
    """Return Hessian contributions for a pair of ARIMA-related parameters.

    The returned tuple contains the contribution through the variance equation
    and the direct contribution through the error equation.
    """
    de2_zeta_xi_RE, de2_zeta_xi = dd_func_lags_mult_arima(
        panel, ll, g, AMAL, vname1, vname2, transpose, u_gradient
    )
    dd_re_variance = dd_func_re_variance(
        panel, ll, g, vname1, vname2, de2_zeta_xi_RE, u_gradient
    )
    return dd_func_garch(
        panel,
        ll,
        g,
        vname1,
        vname2,
        de2_zeta_xi_RE,
        de2_zeta_xi,
        dd_re_variance,
        u_gradient,
    )


def dd_func_lags_mult_arima(
    panel, ll, g, AMAL, vname1, vname2, transpose, u_gradient
):
    """Second derivative of the ARIMA error term.

    ``vname1`` is xi and ``vname2`` is zeta in the mathematical notation.  The
    result is an ``(N, T, m, k)``-style object, later summed into a Hessian block.
    """
    de_xi = _grad(g, vname1)
    de_zeta = _grad(g, vname2)

    if de_xi is None or de_zeta is None:
        return None, None

    if AMAL is None:
        return None, None

    if u_gradient:
        # For the error beta-rho covariance, the u-gradient must be used.
        de2_zeta_xi = -fu.arma_dot(AMAL, g.X_RE, ll)
    else:
        de2_zeta_xi = fu.arma_dot(AMAL, de_zeta, ll)

    if transpose:
        # Used for diagonal lag blocks; add the symmetric counterpart.
        de2_zeta_xi = de2_zeta_xi + np.swapaxes(de2_zeta_xi, 2, 3)

    de2_zeta_xi = de2_zeta_xi * panel.included[4]
    return de2_zeta_xi, de2_zeta_xi


def dd_func_re_variance(panel, ll, g, vname1, vname2, de2_zeta_xi_RE, u_gradient):
    """Second derivative of the random-effect variance component."""
    if panel.N <= 1 or panel.options.fixed_random_group_eff == 0:
        return None

    de_xi_RE = _grad(g, vname1)
    de_zeta_RE = _grad(g, vname2)
    if de_xi_RE is None or de_zeta_RE is None:
        return None

    N, T, m = de_xi_RE.shape
    _, _, k = de_zeta_RE.shape
    incl = panel.included[4]

    de_xi_RE = de_xi_RE.reshape(N, T, m, 1)
    de_zeta_RE = de_zeta_RE.reshape(N, T, 1, k)
    dvRE_xi = _dvRE(g, vname1).reshape(N, T, m, 1)
    dvRE_zeta = _dvRE(g, vname2).reshape(N, T, 1, k)

    if de2_zeta_xi_RE is None:
        ddvRE_d_xi_zeta = None
    else:
        # d²(e_RE²)/(dxi dzeta) = 2 de_xi de_zeta + 2 e_RE d²e/(dxi dzeta)
        dd_e_RE_sq = (
            2 * de_xi_RE * de_zeta_RE
            + 2 * ll.e_RE.reshape(N, T, 1, 1) * de2_zeta_xi_RE
        ) * incl
        ddvRE_d_xi_zeta = panel.mean(dd_e_RE_sq, (0, 1)) * incl

    dvarRE = ll.dvarRE.reshape(N, T, 1, 1)
    ddvarRE = ll.ddvarRE.reshape(N, T, 1, 1)
    return add((prod((dvarRE, ddvRE_d_xi_zeta)), ddvarRE * dvRE_xi * dvRE_zeta))


def dd_func_garch(
    panel,
    ll,
    g,
    vname1,
    vname2,
    de2_zeta_xi_RE,
    de2_zeta_xi,
    dd_re_variance,
    u_gradient,
):
    """Second derivative contribution through the GARCH variance recursion."""
    incl = panel.included[4]
    de_xi_RE = _grad(g, vname1)
    de_zeta_RE = _grad(g, vname2)

    if de_xi_RE is None or de_zeta_RE is None:
        return None, None

    N, T, m = de_xi_RE.shape
    _, _, k = de_zeta_RE.shape

    d2LL_d2e_zeta_xi_RE = None
    if de2_zeta_xi_RE is not None:
        dLL_e = g.dLL_e.reshape(N, T, 1, 1)
        d2LL_d2e_zeta_xi_RE = np.sum(de2_zeta_xi_RE * dLL_e * incl, axis=(0, 1))

    d2var_zeta_xi_h = None
    if panel.pqdkm[4] > 0:
        de2 = 0 if de2_zeta_xi_RE is None else de2_zeta_xi_RE
        h_e_de2 = ll.h_e_val.reshape(N, T, 1, 1) * de2
        h_2e_cross = (
            ll.h_2e_val.reshape(N, T, 1, 1)
            * de_xi_RE.reshape(N, T, m, 1)
            * de_zeta_RE.reshape(N, T, 1, k)
        )
        d2var_zeta_xi_h = fu.arma_dot(ll.GAR_1MA, h_e_de2 + h_2e_cross, ll)

    d2var_zeta_xi = add((d2var_zeta_xi_h, dd_re_variance), ignore=True)
    if d2var_zeta_xi is None:
        return None, d2LL_d2e_zeta_xi_RE

    dLL_var = g.dLL_var.reshape(N, T, 1, 1)
    d2LL_d2var_zeta_xi = np.sum(prod((d2var_zeta_xi, dLL_var * incl), True), axis=(0, 1))
    return d2LL_d2var_zeta_xi, d2LL_d2e_zeta_xi_RE


def dd_func_lags(panel, ll, L, d, dLL, transpose=False):
    """Second derivative contribution from a lag polynomial.

    Parameters
    ----------
    L : tuple, ndarray, or None
        Lag representation.  Tuples are passed to ``fu.arma_dot``; two-
        dimensional arrays are applied directly with ``fu.dot``.
    d : ndarray or None
        First derivative, shape ``(N, T, m)``.
    dLL : ndarray
        Likelihood derivative, shape compatible with ``(N, T)``.
    """
    if panel.pqdkm[4] == 0 or d is None:
        return None

    N, T, m = d.shape
    if L is None:
        x = 0
    elif len(L) == 0:
        return None
    elif isinstance(L, tuple):
        x = fu.arma_dot(L, d, ll)
        if x is None:
            return None
    elif len(L.shape) == 2:
        x = fu.dot(L, d).reshape(N, T, 1, m)
    else:
        raise ValueError("L must be None, a tuple, or a two-dimensional array")

    dLL = dLL.reshape(N, T, 1, 1)
    return np.sum(_clip_extreme(dLL) * _clip_extreme(x), axis=(0, 1))


def dd_func_z(ll, d_arma):
    """Second derivative contribution for exogenous variance terms ``z``."""
    if d_arma is None:
        return None
    x = fu.arma_dot(ll.GAR_1MA, ll.h_ez_val, ll)
    _, _, k = d_arma.shape
    return np.sum(prod((x, d_arma)), axis=(0, 1)).reshape(1, k)


def add(iterable, ignore=False):
    """Add elements, with optional skipping of ``None`` values.

    If ``ignore`` is false, the first ``None`` makes the whole result ``None``.
    """
    result = None
    for value in iterable:
        if value is None:
            if not ignore:
                return None
            continue
        result = value if result is None else result + value
    return result


def prod(iterable, ignore=False):
    """Multiply elements, with optional skipping of ``None`` values."""
    result = None
    for value in iterable:
        if value is None:
            if not ignore:
                return None
            continue
        result = value if result is None else _clip_extreme(result) * _clip_extreme(value)
    return result


def sumNT(nparray):
    """Sum over the ``N`` and ``T`` dimensions while preserving block axes."""
    if nparray is None:
        return None
    if nparray.ndim < 3:
        raise RuntimeError("Not enough dimensions")
    return np.sum(nparray.reshape(list(nparray.shape) + [1]), axis=(0, 1))


def concat_matrix(block_matrix):
    """Concatenate a nested Hessian block matrix, ignoring absent blocks."""
    rows = []
    for row in block_matrix:
        present_blocks = [block for block in row if block is not None]
        if present_blocks:
            rows.append(np.concatenate(present_blocks, axis=1))
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def concat_marray(matrix_array):
    """Concatenate derivative arrays along the parameter axis."""
    present_arrays = [arr for arr in matrix_array if arr is not None]
    if not present_arrays:
        return None
    return np.concatenate(present_arrays, axis=2)


def dd_func(
    d2LL_de2,
    d2LL_dln_de,
    d2LL_dln2,
    de_dh,
    de_dg,
    dln_dh,
    dln_dg,
    dLL_de2_dh_dg,
    dLL_dln2_dh_dg,
):
    """Combine all terms for one Hessian block.

    This implements the chain-rule expansion for two parameter blocks ``h`` and
    ``g``, combining error-equation and log-variance-equation derivatives.
    """
    terms = (
        dd_func_mult(de_dh, d2LL_de2, de_dg),
        dd_func_mult(de_dh, d2LL_dln_de, dln_dg),
        dd_func_mult(dln_dh, d2LL_dln_de, de_dg),
        dd_func_mult(dln_dh, d2LL_dln2, dln_dg),
        dLL_de2_dh_dg,
        dLL_dln2_dh_dg,
    )
    return add(terms, ignore=True)


def dd_func_mult(d0, mult, d1):
    """Return ``sum_NT d0 * mult * d1`` as a Hessian block.

    ``d0`` has shape ``(N, T, k)`` and ``d1`` has shape ``(N, T, m)``.  The
    result has shape ``(k, m)``.
    """
    if d0 is None or d1 is None or mult is None:
        return None

    N, T, k = d0.shape
    _, _, m = d1.shape

    if np.any(np.isnan(d0)) or np.any(np.isnan(d1)):
        out = np.empty((k, m))
        out[:] = np.nan
        return out

    d0 = (_clip_extreme(d0) * mult).reshape(N, T, k, 1)
    d1 = _clip_extreme(d1).reshape(N, T, 1, m)
    return np.sum(d0 * d1, axis=(0, 1))
