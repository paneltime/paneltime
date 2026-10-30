"""Panel working regression and ARMA/GARCH starting values."""

import numpy as np

from ..output import stat_functions as stat
from ..random_effects import REObj, fit_group_effects_stationary
from ..likelihood.function import LLFunction
from . import arma_start


def start_values(panel):
    if _use_static_start(panel):
        return static_start(panel)
    p, q, _, k, m = panel.pqdkm
    included = panel.included[3]
    if p or q:
        beta, ar, ma, innovations = arma_start.fit_arma(panel)
        residuals = innovations[:, :, None]
        squared_correlations = stat.correlogram(
            panel, residuals*residuals, 2, center=True,
        )[1:]
        variance = np.var(innovations[arma_start.observed_mask(panel)], ddof=1)
    else:
        _, _, beta, residuals = arma_start.working_regression(panel)
        ar, ma = np.zeros(p), np.zeros(q)
        correlations = stat.correlogram(panel, residuals, 2, center=True)[1:]
        squared_correlations = stat.correlogram(
            panel, residuals*residuals*included, 2, center=True,
        )[1:] - correlations**2
        variance = panel.var(residuals)
    arch, garch = GARCH_process_calc(squared_correlations, k, m)
    initial_variance = panel.h_func(0, variance, variance)
    omega = initial_variance * (0.2 if panel.options.EGARCH else 1 - arch - garch)
    return beta, ar, ma, arch, garch, variance, initial_variance, omega


def _use_static_start(panel):
    options = panel.options
    return (
        not any(panel.pqdkm)
        and {options.fixed_random_group_eff, options.fixed_random_time_eff} != {1, 2}
        and options.fixed_random_variance_eff == 0
        and options.user_constraints is None
        and options.custom_model is None
        and options.h_function is None
        and not any(limit is not None for limit in options.tobit_limits)
        and panel.nW == 1
        and panel.nZ == 1
    )


def static_start(panel):
    """Static coefficients and profiled variance, before arguments are built."""
    estimates = fit_group_effects_stationary(panel)
    beta = estimates.beta.to_numpy().reshape(-1, 1)
    included = panel.included[3]
    residuals = (panel.Y - np.matmul(panel.X, beta)) * included
    group = REObj(panel, True, panel.T_i, panel.T_i, panel.options.fixed_random_group_eff)
    time = REObj(panel, False, panel.date_count_mtrx, panel.date_count,
                 panel.options.fixed_random_time_eff)
    innovations = (residuals + group.RE(residuals, panel) + time.RE(residuals, panel)) * included
    model = LLFunction(panel, innovations, np.ones_like(innovations), None)
    variance = np.sum(model.e2 * included) / np.sum(included)
    initial_variance = panel.h_func(0, variance, variance)
    return beta, np.zeros(0), np.zeros(0), 0.0, 0.0, variance, initial_variance, initial_variance


def GARCH_process_calc(c, k, m):
    """First ARCH/GARCH starts from squared-innovation autocorrelations.

    For GARCH(1,1), pi=psi+gamma, r2=pi*r1, and
    r1=psi*(1-pi**2+pi*psi)/(1-pi**2+psi**2).
    """
    default = (0.1, 0.5)
    r1, r2 = c
    if m == 0:
        return default
    if k == 0:
        return float(np.clip(r1, 0.01, 0.9)), 0.0
    if r1 < 0.02 or r2 <= 0:
        return default
    pi = np.clip(r2/r1, 0.3, 0.97)
    s = 1 - pi**2
    a = r1 - pi
    if a >= 0:
        return default
    disc = max(s*s - 4*a*r1*s, 0.0)
    psi = np.clip((s - np.sqrt(disc))/(2*a), 0.01, min(0.5, pi - 0.05))
    return float(psi), float(pi - psi)
