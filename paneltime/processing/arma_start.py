"""Full-order ARMA starts: auxiliary regressions and bounded CSS refinement."""

import warnings

import numpy as np

from .. import cfunctions, random_effects as re
from ..output import stat_functions as stat


class ARMAInitializationWarning(RuntimeWarning):
    """A usable preliminary start was returned without full CSS convergence."""


def reflection_coefficients(eta):
    """Stationary AR coefficients and their Jacobian in reflection coordinates."""
    size = len(eta)
    coefficients = np.zeros(size)
    derivative = np.zeros((size, size))
    for j, value in enumerate(np.tanh(eta)):
        previous = coefficients[:j].copy()
        previous_d = derivative[:j].copy()
        coefficients[:j] = previous - value * previous[::-1]
        derivative[:j] = previous_d - value * previous_d[::-1]
        derivative[:j, j] -= (1 - value**2) * previous[::-1]
        coefficients[j] = value
        derivative[j, j] = 1 - value**2
    return coefficients, derivative


def reflection_coordinates(coefficients):
    """Reflect unstable auxiliary roots, then apply the inverse recursion."""
    coefficients = np.asarray(coefficients, dtype=float).copy()
    if not len(coefficients):
        return coefficients
    roots = np.roots(np.r_[1., -coefficients])
    outside = np.abs(roots) > 0.98
    roots[outside] = 0.98**2 / np.conj(roots[outside])
    coefficients = -np.poly(roots).real[1:]
    eta = np.zeros(len(coefficients))
    for j in range(len(coefficients) - 1, -1, -1):
        value = np.clip(coefficients[j], -0.999, 0.999)
        eta[j] = np.arctanh(value)
        coefficients = (
            coefficients[:j] + value * coefficients[:j][::-1]
        ) / (1 - value**2)
    return eta


def run_lengths(included):
    lengths = np.zeros(included.shape, dtype=int)
    for t in range(included.shape[1]):
        lengths[:, t] = included[:, t] * (1 + (lengths[:, t-1] if t else 0))
    return lengths


def lag(values, order, lengths):
    result = np.zeros_like(values)
    if order < values.shape[1]:
        result[:, order:] = values[:, :-order]
    mask = lengths > order
    return result * mask.reshape(mask.shape + (1,) * (values.ndim - 2))


def auxiliary_start(u, p, q, lengths):
    """Pooled Hannan--Rissanen-style regressions, never joining panel runs."""
    n_lags = max(p, q)
    r = min(20, max(2*n_lags + 1, int(np.ceil(np.log(u.shape[1])))))
    while r > 0:
        first = lengths > r
        second = lengths > r + n_lags
        if first.sum() > r and second.sum() > p + q:
            ar_design = np.stack([lag(u, j, lengths) for j in range(1, r+1)], -1)
            ar, _, rank, _ = np.linalg.lstsq(ar_design[first], u[first], rcond=None)
            if rank == r:
                innovations = (u - ar_design @ ar) * first
                columns = [lag(u, j, lengths) for j in range(1, p+1)]
                columns += [lag(innovations, j, lengths) for j in range(1, q+1)]
                design = np.stack(columns, -1)
                coefficients, _, rank, _ = np.linalg.lstsq(
                    design[second], u[second], rcond=None,
                )
                if rank == p + q:
                    return coefficients[:p], coefficients[p:]
        r -= 1
    warnings.warn(
        "ARMA auxiliary regressions are underidentified; using zero ARMA "
        "coefficients to seed the CSS initializer.",
        ARMAInitializationWarning, stacklevel=2,
    )
    return np.zeros(p), np.zeros(q)


def inverse_ma_filter(values, ma, lengths):
    """Batch independent runs/channels through the existing native ARMA filter."""
    included = lengths > 0
    if not len(ma) or not np.any(included):
        return values * included[:, :, None]
    rows, starts = np.nonzero(lengths == 1)
    following = np.zeros_like(included)
    following[:, :-1] = included[:, 1:]
    _, ends = np.nonzero(included & ~following)
    run_sizes = ends - starts + 1
    t_count = int(run_sizes.max())
    channels = values.shape[2]
    packed = np.zeros((len(starts)*channels, t_count))
    for run, (row, start, size) in enumerate(zip(rows, starts, run_sizes)):
        packed[run*channels:(run+1)*channels, :size] = values[row, start:start+size].T
    parameters = np.array([len(packed), t_count, len(ma), 1, 0, 0, 0, 0], dtype=float)
    identity = np.zeros(t_count)
    identity[0] = 1
    filtered = cfunctions.armas(
        parameters, ma, np.ones(1), np.empty(0), np.empty(0),
        identity.copy(), np.zeros(t_count), identity.copy(), np.zeros(t_count),
        packed, np.zeros_like(packed), np.zeros_like(packed), np.zeros_like(packed),
        np.zeros_like(packed), np.repeat(run_sizes, channels), b'',
    )[4]
    if not np.isfinite(filtered).all():
        raise FloatingPointError("Native ARMA initialization filter produced nonfinite values.")
    result = np.zeros_like(values)
    for run, (row, start, size) in enumerate(zip(rows, starts, run_sizes)):
        result[row, start:start+size] = filtered[run*channels:(run+1)*channels, :size].T
    return result


def css_residuals(parameters, x, y, p, q, lengths, jacobian=True):
    """Zero-presample innovations and analytic derivatives, including gaps."""
    n_beta = x.shape[2]
    beta = parameters[:n_beta]
    ar, d_ar = reflection_coefficients(parameters[n_beta:n_beta+p])
    inverse_ma, d_inverse_ma = reflection_coefficients(parameters[n_beta+p:])
    ma, d_ma = -inverse_ma, -d_inverse_ma
    u = (y - x @ beta) * (lengths > 0)
    rhs = u.copy()
    if jacobian:
        direct = np.zeros(x.shape[:2] + (len(parameters),))
        direct[:, :, :n_beta] = -x
    for j in range(p):
        lag_u = lag(u, j+1, lengths)
        rhs -= ar[j] * lag_u
        if jacobian:
            direct[:, :, :n_beta] += ar[j] * lag(x, j+1, lengths)
            direct[:, :, n_beta:n_beta+p] -= lag_u[:, :, None] * d_ar[j]
    e = inverse_ma_filter(rhs[:, :, None], ma, lengths)[:, :, 0]
    derivative = None
    if jacobian:
        for j in range(q):
            direct[:, :, n_beta+p:] -= lag(e, j+1, lengths)[:, :, None] * d_ma[j]
        derivative = inverse_ma_filter(direct, ma, lengths)
    return e, derivative


def working_regression(panel):
    """OLS on the panel's differenced, effects-adjusted working data."""
    re_group = re.REObj(panel, True, panel.T_i, panel.T_i,
                        panel.options.fixed_random_group_eff)
    re_time = re.REObj(panel, False, panel.date_count_mtrx, panel.date_count,
                       panel.options.fixed_random_time_eff)
    included = panel.included[3]
    x = (panel.X + re_group.RE(panel.X, panel) + re_time.RE(panel.X, panel)) * included
    y = (panel.Y + re_group.RE(panel.Y, panel) + re_time.RE(panel.Y, panel)) * included
    beta, residuals = stat.OLS(panel, x, y, return_e=True)
    return x, y, beta, residuals


def observed_mask(panel):
    """Exclude padding and the panel's differencing-boundary observations."""
    observed = panel.included[3][:, :, 0].astype(bool).copy()
    observed[:, :panel.pqdkm[2]] = False
    return observed


def fit_arma(panel):
    """Coordinate the panel working regression, ARMA seed and short CSS fit."""
    p, q = panel.pqdkm[:2]
    x, y, beta, _ = working_regression(panel)
    y = y[:, :, 0]
    included = observed_mask(panel)
    lengths = run_lengths(included)
    selected = lengths > p
    if selected.sum() <= x.shape[2] + p + q:
        warnings.warn(
            "Too few conditional observations for ARMA CSS refinement; "
            "retaining OLS with zero ARMA starts.",
            ARMAInitializationWarning, stacklevel=2,
        )
        return beta, np.zeros(p), np.zeros(q), (y - x @ beta[:, 0]) * included
    x, y, beta, x_scale, y_scale = scale_regression(x, y, beta, included)
    parameters = seed_parameters(x, y, beta, p, q, lengths)
    parameters, innovations = refine_css(parameters, x, y, p, q, lengths)
    return restore_units(parameters, innovations, x_scale, y_scale, p)


def scale_regression(x, y, beta, included):
    """Scale working variables and regression coefficients consistently."""
    x_scale = np.sqrt(np.mean(x[included]**2, axis=0))
    x_scale[x_scale == 0] = 1
    y_scale = np.sqrt(np.mean(y[included]**2))
    if not np.isfinite(y_scale) or y_scale == 0:
        raise ValueError("ARMA initialization requires finite, nonzero working data.")
    x = x / x_scale
    y = y / y_scale
    beta = beta[:, 0] * x_scale / y_scale
    return x, y, beta, x_scale, y_scale


def seed_parameters(x, y, beta, p, q, lengths):
    ar, ma = auxiliary_start((y - x @ beta) * (lengths > 0), p, q, lengths)
    return np.r_[beta, reflection_coordinates(ar), reflection_coordinates(-ma)]


def css_step(parameters, objective, gradient, hessian, damping, x, y, p, q, lengths):
    """Find an objective-decreasing step and adapt the damping to its quality."""
    selected = lengths > p
    diagonal = np.maximum(np.diag(hessian), 1e-8)
    for _ in range(10):
        step = np.linalg.solve(hessian + damping*np.diag(diagonal), -gradient)
        candidate = parameters + step
        # Keep numerical reflection coefficients away from exactly +/-1.
        candidate[x.shape[2]:] = np.clip(candidate[x.shape[2]:], -4, 4)
        trial, _ = css_residuals(candidate, x, y, p, q, lengths, jacobian=False)
        trial_objective = np.mean(trial[selected]**2)
        if np.isfinite(trial_objective) and trial_objective < objective:
            improvement = objective - trial_objective
            realized_step = candidate - parameters
            predicted = -2*gradient @ realized_step - realized_step @ hessian @ realized_step
            ratio = improvement / predicted if predicted > 0 else 0
            if ratio < 0.25:
                damping *= 10
            elif ratio > 0.75:
                damping = max(damping/3, 1e-9)
            return candidate, trial_objective, damping
        damping *= 10
    return None


def refine_css(parameters, x, y, p, q, lengths):
    """Bounded Gauss--Newton refinement of the scaled starting vector."""
    selected = lengths > p
    damping = 1e-3
    e, derivative = css_residuals(parameters, x, y, p, q, lengths)
    objective = np.mean(e[selected]**2)
    for _ in range(10):
        if derivative is None:
            raise RuntimeError("CSS derivatives were not calculated.")
        design = derivative[selected]
        gradient = design.T @ e[selected] / selected.sum()
        hessian = design.T @ design / selected.sum()
        if np.max(np.abs(gradient)) <= 1e-4:
            break
        accepted = css_step(
            parameters, objective, gradient, hessian, damping, x, y, p, q, lengths,
        )
        if accepted is None:
            warnings.warn(
                "ARMA CSS initializer stalled; retaining its best accepted start.",
                ARMAInitializationWarning, stacklevel=2,
            )
            break
        candidate, trial_objective, damping = accepted
        improvement = objective - trial_objective
        parameters, objective = candidate, trial_objective
        e, derivative = css_residuals(parameters, x, y, p, q, lengths)
        if improvement <= 1e-4 * max(objective, 1e-12):
            break
    else:
        warnings.warn(
            "ARMA CSS initializer reached 10 iterations; retaining its best "
            "accepted start.",
            ARMAInitializationWarning, stacklevel=2,
        )
    return parameters, e


def restore_units(parameters, innovations, x_scale, y_scale, p):
    n_beta = len(x_scale)
    beta = (parameters[:n_beta] * y_scale / x_scale)[:, None]
    ar = reflection_coefficients(parameters[n_beta:n_beta+p])[0]
    ma = -reflection_coefficients(parameters[n_beta+p:])[0]
    return beta, ar, ma, innovations * y_scale
