#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Internal stationary FE/RE starting estimators used by processing.initreg.

This module reuses the existing panel object and random-effects helpers to
compute panel estimators in closed form when order=(0,0,0) and
garch_order=(0,0).

Implemented paths:
- pooled OLS (no effects)
- one-way and two-way fixed effects (group/time)
- one-way random effects (group or time)
- two-way random effects (group + time)

Eligibility, including exclusion of mixed fixed/random effects, is decided
by processing.initreg before calling this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dynamic import REObj


_EFFECT_MAP = {0: "none", 1: "fixed", 2: "random"}


@dataclass
class StationaryEffectsResult:
    group_mode: str
    time_mode: str
    beta: pd.Series
    residuals: np.ndarray
    fitted: np.ndarray
    group_effects: pd.Series
    time_effects: pd.Series
    sigma_e2: float
    sigma_alpha2: float
    sigma_lambda2: float
    theta_group: pd.Series
    theta_time: pd.Series
    nobs: int


class StationaryEffectsError(RuntimeError):
    pass


# Shared numeric helpers
def _safe_lstsq(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _fitted_from_beta(X, beta):
    return np.tensordot(X, beta, axes=([2], [0]))[:, :, None]


@dataclass
class _Layout:
    """Group and date of every cell in the (N, T) panel arrays.

    Each group's observations are packed from column 0, so a column is a
    position within the group, not a calendar date. Time effects are therefore
    indexed by ``panel.date_map``, as in ``dynamic.py``.
    """

    mask: np.ndarray
    index: dict
    n_levels: dict

    def obs_index(self, axis):
        """Group or date index of each observed cell, in ``mask`` order."""
        return self.index[axis][self.mask]


def _layout(panel, mask):
    n_groups, n_cols = mask.shape
    dates = np.zeros((n_groups, n_cols), dtype=int)
    for date, cells in enumerate(panel.date_map):
        dates[cells] = date
    groups = np.broadcast_to(np.arange(n_groups)[:, None], (n_groups, n_cols))
    return _Layout(
        mask=mask,
        index={"group": groups, "time": dates},
        n_levels={"group": n_groups, "time": len(panel.date_map)},
    )


def _counts(layout, axis):
    return np.bincount(layout.obs_index(axis), minlength=layout.n_levels[axis]).astype(float)


def _masked_axis_means(values, idx, n_levels):
    """Mean of observation-level values by index levels.

    values: (nobs, p), idx: (nobs,)
    """
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    sums = np.zeros((n_levels, values.shape[1]), dtype=float)
    counts = np.bincount(idx, minlength=n_levels).astype(float)
    for j in range(values.shape[1]):
        sums[:, j] = np.bincount(idx, weights=values[:, j], minlength=n_levels)
    means = np.zeros_like(sums)
    valid = counts > 0
    if np.any(valid):
        means[valid] = sums[valid] / counts[valid, None]
    return means, counts, valid


def _level_means(layout, values, axis):
    """Masked means of (N, T, k) values per group or date, shaped (levels, k)."""
    means, _, _ = _masked_axis_means(
        values[layout.mask], layout.obs_index(axis), layout.n_levels[axis]
    )
    return means


def _expand(layout, level_values, axis):
    """Place per-group or per-date values on the cells, shaped (N, T, 1) or (N, T, k)."""
    level_values = np.asarray(level_values, dtype=float)
    out = level_values[layout.index[axis]]
    return out[:, :, None] if level_values.ndim == 1 else out


def _flatten_masked(X, y, mask):
    X_obs = X[mask, :]
    y_obs = y[:, :, 0][mask]
    return X_obs, y_obs


def _beta_names(panel):
    return list(panel.input.X.columns)


def _center_effects(values, counts, has_intercept):
    if not has_intercept:
        return values, 0.0
    total = float(np.sum(counts))
    if total <= 0:
        return values, 0.0
    weighted_mean = float(np.sum(values * counts) / total)
    return values - weighted_mean, weighted_mean


def _build_result(
    beta,
    beta_names,
    residuals,
    fitted,
    group_effects,
    time_effects,
    sigma_e2,
    sigma_alpha2,
    sigma_lambda2,
    theta_group,
    theta_time,
    nobs,
    group_mode,
    time_mode,
):
    return StationaryEffectsResult(
        group_mode=group_mode,
        time_mode=time_mode,
        beta=pd.Series(np.asarray(beta, dtype=float), index=beta_names, dtype=float),
        residuals=residuals,
        fitted=fitted,
        group_effects=pd.Series(np.asarray(group_effects, dtype=float)),
        time_effects=pd.Series(np.asarray(time_effects, dtype=float)),
        sigma_e2=float(sigma_e2),
        sigma_alpha2=float(sigma_alpha2),
        sigma_lambda2=float(sigma_lambda2),
        theta_group=pd.Series(np.asarray(theta_group, dtype=float)),
        theta_time=pd.Series(np.asarray(theta_time, dtype=float)),
        nobs=int(nobs),
    )


# Pooled (no-effects) stationary fit
def _fit_none(panel, X, y, layout, beta_names):
    mask = layout.mask
    X_obs, y_obs = _flatten_masked(X, y, mask)
    beta = _safe_lstsq(X_obs, y_obs)
    fitted = _fitted_from_beta(X, beta)
    residuals = (y - fitted) * panel.included[3]
    sigma_e2 = float(np.sum(residuals[:, :, 0][mask] ** 2) / max(mask.sum() - X.shape[2], 1))
    n_groups, n_times = layout.n_levels["group"], layout.n_levels["time"]

    return _build_result(
        beta=beta,
        beta_names=beta_names,
        residuals=residuals,
        fitted=fitted,
        group_effects=np.zeros(n_groups),
        time_effects=np.zeros(n_times),
        sigma_e2=sigma_e2,
        sigma_alpha2=0.0,
        sigma_lambda2=0.0,
        theta_group=np.zeros(n_groups),
        theta_time=np.zeros(n_times),
        nobs=int(mask.sum()),
        group_mode="none",
        time_mode="none",
    )


def _fixed_keep_columns(panel, n_columns):
    keep = np.ones(n_columns, dtype=bool)
    if bool(panel.input.has_intercept):
        keep[0] = False
    return keep


def _one_way_within(values, idx, n_levels):
    means, _, _ = _masked_axis_means(values, idx, n_levels)
    return values - means[idx]


def _two_way_within(values, g_idx, t_idx, n_groups, n_times, max_iter=200, tol=1e-10):
    """Alternating demeaning for unbalanced two-way FE residualization."""
    z = np.asarray(values, dtype=float).copy()
    for _ in range(max_iter):
        prev = z.copy()
        g_means, _, _ = _masked_axis_means(z, g_idx, n_groups)
        z = z - g_means[g_idx]
        t_means, _, _ = _masked_axis_means(z, t_idx, n_times)
        z = z - t_means[t_idx]
        if np.max(np.abs(z - prev)) < tol:
            break
    return z


def _recover_effects_from_residuals(layout, residual_obs, group_fixed, time_fixed):
    """Recover additive FE components from observation residuals."""
    g_idx, t_idx = layout.obs_index("group"), layout.obs_index("time")
    n_groups, n_times = layout.n_levels["group"], layout.n_levels["time"]

    if group_fixed and not time_fixed:
        group_means, _, _ = _masked_axis_means(residual_obs, g_idx, n_groups)
        return group_means[:, 0], np.zeros(n_times, dtype=float)

    if time_fixed and not group_fixed:
        time_means, _, _ = _masked_axis_means(residual_obs, t_idx, n_times)
        return np.zeros(n_groups, dtype=float), time_means[:, 0]

    if not group_fixed and not time_fixed:
        return np.zeros(n_groups, dtype=float), np.zeros(n_times, dtype=float)

    return _additive_blup(residual_obs, g_idx, t_idx, np.ones(n_groups), np.ones(n_times))


# Fixed-effects stationary fit (one-way and two-way)
def _fit_fixed_effects(panel, X, y, layout, beta_names, group_fixed, time_fixed):
    if not group_fixed and not time_fixed:
        return _fit_none(panel, X, y, layout, beta_names)

    X_obs, y_obs = _flatten_masked(X, y, layout.mask)
    keep = _fixed_keep_columns(panel, X.shape[2])
    g_idx, t_idx = layout.obs_index("group"), layout.obs_index("time")
    n_groups, n_times = layout.n_levels["group"], layout.n_levels["time"]

    X_work = X_obs[:, keep]
    y_work = y_obs.reshape((-1, 1))

    if group_fixed and time_fixed:
        X_tilde = _two_way_within(X_work, g_idx, t_idx, n_groups, n_times)
        y_tilde = _two_way_within(y_work, g_idx, t_idx, n_groups, n_times)[:, 0]
    elif group_fixed:
        X_tilde = _one_way_within(X_work, g_idx, n_groups)
        y_tilde = _one_way_within(y_work, g_idx, n_groups)[:, 0]
    else:
        X_tilde = _one_way_within(X_work, t_idx, n_times)
        y_tilde = _one_way_within(y_work, t_idx, n_times)[:, 0]

    beta = np.zeros(X.shape[2], dtype=float)
    if X_tilde.shape[1] > 0:
        beta[keep] = _safe_lstsq(X_tilde, y_tilde)

    residual_obs = y_obs - X_work @ beta[keep]
    group_effects, time_effects = _recover_effects_from_residuals(
        layout, residual_obs, group_fixed=group_fixed, time_fixed=time_fixed
    )
    beta, group_effects, time_effects = _fixed_adjust_intercept(
        panel,
        layout,
        X_obs,
        y_obs,
        beta,
        keep,
        group_effects,
        time_effects,
        group_fixed,
        time_fixed,
    )
    fitted, residuals, sigma_e2, nobs = _fixed_fitted_and_sigma(
        panel,
        X,
        y,
        layout,
        beta,
        group_effects,
        time_effects,
        group_fixed,
        time_fixed,
        keep,
    )

    return _build_result(
        beta=beta,
        beta_names=beta_names,
        residuals=residuals,
        fitted=fitted,
        group_effects=group_effects,
        time_effects=time_effects,
        sigma_e2=sigma_e2,
        sigma_alpha2=0.0,
        sigma_lambda2=0.0,
        theta_group=np.ones_like(group_effects) if group_fixed else np.zeros_like(group_effects),
        theta_time=np.ones_like(time_effects) if time_fixed else np.zeros_like(time_effects),
        nobs=nobs,
        group_mode="fixed" if group_fixed else "none",
        time_mode="fixed" if time_fixed else "none",
    )


def _fixed_adjust_intercept(
    panel,
    layout,
    X_obs,
    y_obs,
    beta,
    keep,
    group_effects,
    time_effects,
    group_fixed,
    time_fixed,
):
    if not bool(panel.input.has_intercept):
        return beta, group_effects, time_effects

    if group_fixed:
        group_effects, _ = _center_effects(group_effects, _counts(layout, "group"), True)
    if time_fixed:
        time_effects, _ = _center_effects(time_effects, _counts(layout, "time"), True)

    # The effects are centered, so the mean remaining residual is the intercept.
    base = X_obs[:, keep] @ beta[keep]
    if group_fixed:
        base = base + group_effects[layout.obs_index("group")]
    if time_fixed:
        base = base + time_effects[layout.obs_index("time")]
    beta[0] = float(np.mean(y_obs - base))
    return beta, group_effects, time_effects


def _fixed_fitted_and_sigma(
    panel,
    X,
    y,
    layout,
    beta,
    group_effects,
    time_effects,
    group_fixed,
    time_fixed,
    keep,
):
    fitted = _fitted_from_beta(X, beta)
    if group_fixed:
        fitted = fitted + _expand(layout, group_effects, "group")
    if time_fixed:
        fitted = fitted + _expand(layout, time_effects, "time")
    residuals = (y - fitted) * panel.included[3]

    mask = layout.mask
    nobs = int(mask.sum())
    n_groups, n_times = layout.n_levels["group"], layout.n_levels["time"]
    effect_df = int(group_fixed) * (n_groups - 1) + int(time_fixed) * (n_times - 1)
    slope_df = int(np.sum(keep))
    df_e = max(nobs - effect_df - slope_df - int(panel.input.has_intercept), 1)
    sigma_e2 = float(np.sum(residuals[:, :, 0][mask] ** 2) / df_e)
    return fitted, residuals, sigma_e2, nobs


def _one_way_fe_sigma(panel, X, y, layout, beta_names, axis):
    fe = _fit_fixed_effects(
        panel,
        X,
        y,
        layout,
        beta_names,
        group_fixed=(axis == "group"),
        time_fixed=(axis == "time"),
    )
    return float(max(fe.sigma_e2, 0.0))


def _one_way_between_components(panel, X, y, layout, axis):
    X_b = _level_means(layout, X, axis)
    y_b = _level_means(layout, y, axis)[:, 0]
    counts = _counts(layout, axis)
    valid = counts > 0
    mean_X = _expand(layout, X_b, axis)
    mean_y = _expand(layout, y_b, axis)
    return mean_X, mean_y, X_b, y_b, counts, valid


def _one_way_variance_component(X_b, y_b, counts, valid, sigma_e2):
    X_bw = X_b[valid]
    y_bw = y_b[valid]
    w = counts[valid]
    w_sqrt = np.sqrt(w)
    beta_bw = _safe_lstsq(X_bw * w_sqrt[:, None], y_bw * w_sqrt)
    resid_bw = y_bw - X_bw @ beta_bw
    rank_bw = np.linalg.matrix_rank(X_bw)
    df_bw = max(int(np.sum(valid)) - rank_bw, 1)
    X_weighted = X_bw * w_sqrt[:, None]
    leverage = np.sum(X_weighted * np.linalg.pinv(X_weighted).T, axis=1)
    # E[weighted RSS] = sigma_e2 * residual_df + sigma_axis * effect_trace.
    effect_trace = float(np.sum(w * (1.0 - leverage)))
    if effect_trace <= 0:
        raise StationaryEffectsError("Between regression has no degrees of freedom for random effects")
    return float(max(0.0, (np.sum(w * resid_bw**2) - sigma_e2 * df_bw) / effect_trace))


def _one_way_theta(counts, valid, sigma_e2, sigma_axis):
    theta = np.zeros(len(counts), dtype=float)
    if sigma_axis > 0:
        denom = sigma_e2 + counts * sigma_axis
        theta[valid] = 1.0 - np.sqrt(np.maximum(sigma_e2 / denom[valid], 0.0))
    return theta


def _one_way_beta(X, y, layout, axis, mean_X, mean_y, theta):
    theta_mtx = _expand(layout, theta, axis)
    X_star = X - theta_mtx * mean_X
    y_star = y - theta_mtx * mean_y
    mask = layout.mask
    return _safe_lstsq(X_star[mask, :], y_star[:, :, 0][mask])


def _one_way_effects(panel, X, y, layout, axis, beta, counts, valid, sigma_e2, sigma_axis):
    fitted_common = _fitted_from_beta(X, beta)
    mu_axis = _level_means(layout, y - fitted_common, axis)[:, 0]

    effects_axis = np.zeros_like(mu_axis)
    if sigma_axis > 0:
        shrink = np.zeros_like(mu_axis)
        shrink[valid] = sigma_axis / (sigma_axis + sigma_e2 / counts[valid])
        effects_axis = shrink * mu_axis

    effects_axis, shift = _center_effects(effects_axis, counts, panel.input.has_intercept)
    if panel.input.has_intercept and len(beta) > 0:
        beta[0] += shift
    return beta, fitted_common, effects_axis


def _one_way_mode_payload(axis, layout, effects_axis, sigma_axis, theta):
    n_groups, n_times = layout.n_levels["group"], layout.n_levels["time"]
    if axis == "group":
        return {
            "group_effects": effects_axis,
            "time_effects": np.zeros(n_times),
            "sigma_alpha2": sigma_axis,
            "sigma_lambda2": 0.0,
            "theta_group": theta,
            "theta_time": np.zeros(n_times),
            "group_mode": "random",
            "time_mode": "none",
        }
    return {
        "group_effects": np.zeros(n_groups),
        "time_effects": effects_axis,
        "sigma_alpha2": 0.0,
        "sigma_lambda2": sigma_axis,
        "theta_group": np.zeros(n_groups),
        "theta_time": theta,
        "group_mode": "none",
        "time_mode": "random",
    }


# One-way random-effects stationary fit
def _fit_random_one_way(panel, X, y, layout, beta_names, axis):
    sigma_e2 = _one_way_fe_sigma(panel, X, y, layout, beta_names, axis)
    mean_X, mean_y, X_b, y_b, counts, valid = _one_way_between_components(
        panel,
        X,
        y,
        layout,
        axis,
    )
    sigma_axis = _one_way_variance_component(X_b, y_b, counts, valid, sigma_e2)
    theta = _one_way_theta(counts, valid, sigma_e2, sigma_axis)
    beta = _one_way_beta(X, y, layout, axis, mean_X, mean_y, theta)
    beta, fitted_common, effects_axis = _one_way_effects(
        panel, X, y, layout, axis, beta, counts, valid, sigma_e2, sigma_axis
    )

    fitted = fitted_common + _expand(layout, effects_axis, axis)
    residuals = (y - fitted) * panel.included[3]
    mode_payload = _one_way_mode_payload(axis, layout, effects_axis, sigma_axis, theta)

    return _build_result(
        beta=beta,
        beta_names=beta_names,
        residuals=residuals,
        fitted=fitted,
        group_effects=mode_payload["group_effects"],
        time_effects=mode_payload["time_effects"],
        sigma_e2=sigma_e2,
        sigma_alpha2=mode_payload["sigma_alpha2"],
        sigma_lambda2=mode_payload["sigma_lambda2"],
        theta_group=mode_payload["theta_group"],
        theta_time=mode_payload["theta_time"],
        nobs=int(layout.mask.sum()),
        group_mode=mode_payload["group_mode"],
        time_mode=mode_payload["time_mode"],
    )


def _additive_blup(res_obs, g_idx, t_idx, w_g, w_t, max_iter=1000, tol=1e-10):
    """Alternating shrunken group/date means of observation residuals."""
    n_groups, n_times = len(w_g), len(w_t)
    alpha = np.zeros(n_groups, dtype=float)
    lam = np.zeros(n_times, dtype=float)
    counts_g = np.bincount(g_idx, minlength=n_groups).astype(float)
    counts_t = np.bincount(t_idx, minlength=n_times).astype(float)
    valid_g = counts_g > 0
    valid_t = counts_t > 0

    for _ in range(max_iter):
        prev_alpha = alpha.copy()
        prev_lam = lam.copy()

        sums_g = np.bincount(g_idx, weights=res_obs - lam[t_idx], minlength=n_groups)
        alpha[valid_g] = w_g[valid_g] * sums_g[valid_g] / counts_g[valid_g]
        alpha -= float(np.mean(alpha))

        sums_t = np.bincount(t_idx, weights=res_obs - alpha[g_idx], minlength=n_times)
        lam[valid_t] = w_t[valid_t] * sums_t[valid_t] / counts_t[valid_t]
        lam -= float(np.mean(lam))

        if max(float(np.max(np.abs(alpha - prev_alpha))), float(np.max(np.abs(lam - prev_lam)))) < tol:
            break
    return alpha, lam


def _two_way_re_objects(panel, residuals, re_i=None, re_t=None):
    if re_i is None:
        re_i = REObj(panel, True, panel.T_i, panel.T_i, 2)
    if re_t is None:
        re_t = REObj(panel, False, panel.date_count_mtrx, panel.date_count, 2)
    re_i.RE(residuals, panel, recalc=True)
    re_t.RE(residuals, panel, recalc=True)
    return re_i, re_t


def _apply_additive_re_transform(values, panel, incl, re_i, re_t, out):
    """Compute (values + RE_i(values) + RE_t(values)) * incl in-place."""
    values_re_i = re_i.RE(values, panel, recalc=False)
    values_re_t = re_t.RE(values, panel, recalc=False)
    np.add(values, values_re_i, out=out)
    out += values_re_t
    out *= incl
    return out


def _two_way_iterative_beta(panel, X, y, mask, beta, incl, max_iter=50, tol=1e-10):
    beta = np.asarray(beta, dtype=float).copy()
    re_i = REObj(panel, True, panel.T_i, panel.T_i, 2)
    re_t = REObj(panel, False, panel.date_count_mtrx, panel.date_count, 2)
    y_star = np.empty_like(y)
    x_star = np.empty_like(X)
    obs_i, obs_t = np.where(mask)
    residuals = None
    for _ in range(max_iter):
        fitted = _fitted_from_beta(X, beta)
        residuals = (y - fitted) * incl
        re_i, re_t = _two_way_re_objects(panel, residuals, re_i=re_i, re_t=re_t)

        _apply_additive_re_transform(y, panel, incl, re_i, re_t, y_star)
        _apply_additive_re_transform(X, panel, incl, re_i, re_t, x_star)

        beta_new = _safe_lstsq(x_star[obs_i, obs_t, :], y_star[obs_i, obs_t, 0])
        if np.max(np.abs(beta_new - beta)) < tol:
            return beta_new, re_i, re_t, residuals
        beta = beta_new
    return beta, re_i, re_t, residuals


def _two_way_variances_and_weights(layout, re_i, re_t):
    sigma_e2_i = float(np.asarray(re_i.e_var).reshape(-1)[0]) if np.size(re_i.e_var) else 0.0
    sigma_e2_t = float(np.asarray(re_t.e_var).reshape(-1)[0]) if np.size(re_t.e_var) else 0.0
    sigma_e2 = float(max(0.0, 0.5 * (sigma_e2_i + sigma_e2_t)))
    sigma_alpha2 = float(max(0.0, getattr(re_i, "v_var", 0.0)))
    sigma_lambda2 = float(max(0.0, getattr(re_t, "v_var", 0.0)))

    counts_g = _counts(layout, "group")
    counts_t = _counts(layout, "time")
    valid_g = counts_g > 0
    valid_t = counts_t > 0

    w_g = np.zeros_like(counts_g, dtype=float)
    w_t = np.zeros_like(counts_t, dtype=float)
    if sigma_alpha2 > 0:
        w_g[valid_g] = sigma_alpha2 / (sigma_alpha2 + sigma_e2 / counts_g[valid_g])
    if sigma_lambda2 > 0:
        w_t[valid_t] = sigma_lambda2 / (sigma_lambda2 + sigma_e2 / counts_t[valid_t])
    return (
        sigma_e2,
        sigma_alpha2,
        sigma_lambda2,
        counts_g,
        counts_t,
        w_g,
        w_t,
    )


def _two_way_center_intercept(panel, beta, alpha, lam, counts_g, counts_t):
    if not panel.input.has_intercept or len(beta) == 0:
        return beta, alpha, lam
    shift_g = float(np.sum(alpha * counts_g) / max(np.sum(counts_g), 1.0))
    shift_t = float(np.sum(lam * counts_t) / max(np.sum(counts_t), 1.0))
    alpha = alpha - shift_g
    lam = lam - shift_t
    beta = np.asarray(beta, dtype=float).copy()
    beta[0] += shift_g + shift_t
    return beta, alpha, lam


# Two-way random-effects stationary fit
def _fit_random_two_way(panel, X, y, layout, beta_names):
    """Two-way random effects via additive quasi-demeaning and GLS updates.

    The transformation mirrors likelihood/main.py where RE adjustments from
    group and time objects are added additively.
    """
    fe = _fit_fixed_effects(
        panel,
        X,
        y,
        layout,
        beta_names,
        group_fixed=True,
        time_fixed=True,
    )
    mask = layout.mask
    incl = panel.included[3]
    beta, re_i, re_t, _ = _two_way_iterative_beta(panel, X, y, mask, fe.beta, incl)

    fitted_common = _fitted_from_beta(X, beta)
    residuals = (y - fitted_common) * incl
    re_i, re_t = _two_way_re_objects(panel, residuals, re_i=re_i, re_t=re_t)
    sigma_e2, sigma_alpha2, sigma_lambda2, counts_g, counts_t, w_g, w_t = (
        _two_way_variances_and_weights(layout, re_i, re_t)
    )

    alpha, lam = _additive_blup(
        residuals[:, :, 0][mask], layout.obs_index("group"), layout.obs_index("time"), w_g, w_t
    )
    beta, alpha, lam = _two_way_center_intercept(
        panel,
        beta,
        alpha,
        lam,
        counts_g,
        counts_t,
    )

    fitted = fitted_common + _expand(layout, alpha, "group") + _expand(layout, lam, "time")
    e = (y - fitted) * incl

    return _build_result(
        beta=beta,
        beta_names=beta_names,
        residuals=e,
        fitted=fitted,
        group_effects=alpha,
        time_effects=lam,
        sigma_e2=sigma_e2,
        sigma_alpha2=sigma_alpha2,
        sigma_lambda2=sigma_lambda2,
        theta_group=_level_means(layout, re_i.theta, "group")[:, 0],
        theta_time=_level_means(layout, re_t.theta, "time")[:, 0],
        nobs=int(mask.sum()),
        group_mode="random",
        time_mode="random",
    )


def _resolve_effect_modes(panel, mode, time_mode):
    if mode is None:
        mode = _EFFECT_MAP.get(int(panel.options.fixed_random_group_eff), "none")
    if time_mode is None:
        time_mode = _EFFECT_MAP.get(int(panel.options.fixed_random_time_eff), "none")
    return str(mode).lower(), str(time_mode).lower()


# Internal dispatch and entry point
def _dispatch_stationary_fit(panel, X, y, layout, names, mode, time_mode):
    if mode == "none" and time_mode == "none":
        return _fit_none(panel, X, y, layout, names)
    if mode == "fixed" or time_mode == "fixed":
        return _fit_fixed_effects(
            panel,
            X,
            y,
            layout,
            names,
            group_fixed=(mode == "fixed"),
            time_fixed=(time_mode == "fixed"),
        )
    if mode == "random" and time_mode == "none":
        return _fit_random_one_way(panel, X, y, layout, names, axis="group")
    if mode == "none" and time_mode == "random":
        return _fit_random_one_way(panel, X, y, layout, names, axis="time")
    if mode == "random" and time_mode == "random":
        return _fit_random_two_way(panel, X, y, layout, names)
    raise StationaryEffectsError(
        f"Unsupported stationary effect combination: group={mode}, time={time_mode}"
    )


def fit_group_effects_stationary(panel, mode=None, time_mode=None):
    """Estimate stationary FE/RE in closed form from an initialized panel.

    Time effects are returned per calendar date, in the order of
    ``panel.date_map`` (sorted dates).

    Parameters
    ----------
    panel:
        Initialized paneltime processing.panel.Panel object.
    mode:
        Group mode in {'none', 'fixed', 'random'}. If None, it is inferred
        from panel.options.fixed_random_group_eff.
    time_mode:
        Time mode in {'none', 'fixed', 'random'}. If None, it is inferred
        from panel.options.fixed_random_time_eff.
    """
    mode, time_mode = _resolve_effect_modes(panel, mode, time_mode)

    X = np.asarray(panel.X, dtype=float)
    y = np.asarray(panel.Y, dtype=float)
    mask = np.asarray(panel.included[3][:, :, 0], dtype=bool)
    layout = _layout(panel, mask)
    names = _beta_names(panel)
    return _dispatch_stationary_fit(panel, X, y, layout, names, mode, time_mode)
