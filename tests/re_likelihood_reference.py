"""Small, dense Gaussian RE/AR(1) audit reference, not a production estimator.

Run from the package root: python tests\\re_likelihood_reference.py
Expectations use covariance traces, not generated observations.
"""

import json

import numpy as np


def covariance(n, t, rho, innovation_variance, group_variance, time_variance):
    if n < 1 or t < 2 or not isinstance(n, int) or not isinstance(t, int):
        raise ValueError("n and t must be integers with n >= 1 and t >= 2")
    values = (rho, innovation_variance, group_variance, time_variance)
    if not np.isfinite(values).all():
        raise ValueError("All covariance parameters must be finite")
    if abs(rho) >= 1 or innovation_variance <= 0 or min(group_variance, time_variance) < 0:
        raise ValueError("Require |rho| < 1, positive innovation variance and nonnegative RE variances")
    lag = abs(np.arange(t)[:, None] - np.arange(t))
    serial = innovation_variance / (1 - rho**2) * rho**lag
    return (np.kron(np.eye(n), serial)
            + group_variance*np.kron(np.eye(n), np.ones((t, t)))
            + time_variance*np.kron(np.ones((n, n)), np.eye(t)))


def gaussian_loglike(residual, cov):
    residual = np.asarray(residual).reshape(-1)
    factor = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(factor, residual)
    logdet = 2*np.log(np.diag(factor)).sum()
    return float(-0.5*(len(residual)*np.log(2*np.pi) + logdet + whitened@whitened))


def woodbury_loglike(residual, n, t, rho, innovation_variance, group_variance, time_variance):
    """Independent reference: AR innovations plus integrated Gaussian effects."""
    covariance(n, t, rho, innovation_variance, group_variance, time_variance)
    residual = np.asarray(residual).reshape(-1)
    b = np.eye(t) - rho*np.eye(t, k=-1)
    variance = np.full(t, innovation_variance)
    variance[0] /= 1 - rho**2
    precision = np.kron(np.eye(n), b.T @ (b / variance[:, None]))
    logdet = n*np.log(variance).sum()
    columns = []
    if group_variance > 0:
        columns.append(np.sqrt(group_variance)*np.kron(np.eye(n), np.ones((t, 1))))
    if time_variance > 0:
        columns.append(np.sqrt(time_variance)*np.kron(np.ones((n, 1)), np.eye(t)))
    quadratic = residual@precision@residual
    if columns:
        z = np.concatenate(columns, axis=1)
        small = np.eye(z.shape[1]) + z.T@precision@z
        factor = np.linalg.cholesky(small)
        logdet += 2*np.log(np.diag(factor)).sum()
        score = z.T@precision@residual
        quadratic -= score@np.linalg.solve(small, score)
    return float(-0.5*(n*t*np.log(2*np.pi) + logdet + quadratic))


def quasi_demeaning(n, t, theta_group, theta_time):
    grand = np.ones((n*t, n*t))/(n*t)
    group = np.kron(np.eye(n), np.ones((t, t))/t) - grand
    time = np.kron(np.ones((n, n))/n, np.eye(t)) - grand
    return np.eye(n*t) - theta_group*group - theta_time*time


def transformed_variance(n, t, rho, innovation_variance, first_variance):
    b = np.kron(np.eye(n), np.eye(t) - rho*np.eye(t, k=-1))
    variance = np.full((n, t), innovation_variance)
    variance[:, 0] = first_variance
    return b, variance.reshape(-1)


def expected_exact_loglike(candidate, true_cov):
    factor = np.linalg.cholesky(candidate)
    logdet = 2*np.log(np.diag(factor)).sum()
    return float(-0.5*(len(candidate)*np.log(2*np.pi) + logdet
                       + np.trace(np.linalg.solve(candidate, true_cov))))


def expected_quasi_loglike(n, t, rho, innovation_variance, first_variance, a, true_cov):
    b, variance = transformed_variance(n, t, rho, innovation_variance, first_variance)
    w = b@a
    diagonal = np.diag(w@true_cov@w.T)
    return float(-0.5*(n*t*np.log(2*np.pi) + np.log(variance).sum()
                       + (diagonal/variance).sum()))


def benchmark_case(n=4, t=8, rho=0.5, innovation_variance=2., group_variance=1., time_variance=1.):
    true_cov = covariance(n, t, rho, innovation_variance, group_variance, time_variance)
    idiosyncratic_variance = innovation_variance/(1 - rho**2)
    theta_g = 1 - np.sqrt(idiosyncratic_variance/(idiosyncratic_variance + t*group_variance))
    theta_t = 1 - np.sqrt(idiosyncratic_variance/(idiosyncratic_variance + n*time_variance))
    a = quasi_demeaning(n, t, theta_g, theta_t)
    b, variance = transformed_variance(n, t, rho, innovation_variance, idiosyncratic_variance)
    whitened_cov = ((b@a@true_cov@a.T@b.T)
                    / np.sqrt(np.outer(variance, variance)))
    grand_scale = np.sqrt(idiosyncratic_variance/(
        idiosyncratic_variance + t*group_variance + n*time_variance))
    adjusted_a = a + (grand_scale - 1)*np.ones((n*t, n*t))/(n*t)
    adjusted_cov = ((b@adjusted_a@true_cov@adjusted_a.T@b.T)
                    / np.sqrt(np.outer(variance, variance)))
    expected_exact = expected_exact_loglike(true_cov, true_cov)
    expected_quasi = expected_quasi_loglike(
        n, t, rho, innovation_variance, idiosyncratic_variance, a, true_cov,
    )
    h = 1e-5
    exact_score, quasi_score = [], []
    for coordinate in range(2):
        exact_values, quasi_values = [], []
        for sign in (-1, 1):
            r = rho + sign*h if coordinate == 0 else rho
            v = innovation_variance*np.exp(sign*h) if coordinate == 1 else innovation_variance
            candidate = covariance(n, t, r, v, group_variance, time_variance)
            exact_values.append(expected_exact_loglike(candidate, true_cov))
            quasi_values.append(expected_quasi_loglike(
                n, t, r, v, idiosyncratic_variance, a, true_cov,
            ))
        exact_score.append((exact_values[1] - exact_values[0])/(2*h))
        quasi_score.append((quasi_values[1] - quasi_values[0])/(2*h))
    return {
        'n': n, 't': t, 'rho': rho, 'innovation_variance': innovation_variance,
        'group_variance': group_variance, 'time_variance': time_variance,
        'theta_group': float(theta_g), 'theta_time': float(theta_t),
        'whitened_covariance_max_error': float(np.max(abs(whitened_cov-np.eye(n*t)))),
        'iid_grand_adjusted_whitening_max_error': float(np.max(abs(adjusted_cov-np.eye(n*t)))),
        'expected_exact_loglike': expected_exact, 'expected_quasi_loglike': expected_quasi,
        'logdet_quasi_demeaning': float(np.linalg.slogdet(a)[1]),
        'exact_expected_score_rho_log_variance': exact_score,
        'frozen_quasi_expected_score_rho_log_variance': quasi_score,
    }


def audit_report():
    return {
        'scope': 'Stationary Gaussian AR(1) plus independent Gaussian group/time effects.',
        'expectation': 'Exact covariance traces; no Monte Carlo or generated observations.',
        'limitation': ('Quasi weights are fixed at theoretical moment values. Expected scores are '
                       'not expectations of the full residual-adaptive paneltime estimator. '
                       'The quasi first-period variance is fixed at the true idiosyncratic level.'),
        'score_coordinates': ['rho', 'log innovation variance'],
        'cases': {
            'iid_no_effects': benchmark_case(rho=0, group_variance=0, time_variance=0),
            'ar1_no_effects': benchmark_case(group_variance=0, time_variance=0),
            'iid_group': benchmark_case(rho=0, time_variance=0),
            'ar1_group': benchmark_case(time_variance=0),
            'iid_two_way': benchmark_case(rho=0),
            'ar1_two_way': benchmark_case(),
        },
    }


if __name__ == '__main__':
    print(json.dumps(audit_report(), indent=2))
