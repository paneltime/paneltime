
**Paneltime** estimates regressions on datasets that:

- Are **panels** (have both a time and a group dimension)
- Are **non-stationary in means** (ARIMA)
- Are **non-stationary in variance** (GARCH)

Paneltime jointly handles common time-series problems (**ARIMA**/**GARCH**) and
panel data with group and time effects. To the best of our knowledge, no other
package combines all three in one model.

The package can also be used on pure **panel** datasets without ARIMA/GARCH or
single time-series data with ARIMA/GARCH problems.

**Author:** Espen Sirnes

**Current version:** 1.3.4

## Installation

Install the package with:

```{python}
#| eval: false
%pip install paneltime
```

## Formula syntax

Model formulas use the form `Y ~ X1 + X2`. Expressions supported by NumPy can
be used with `np` as the alias, for example `np.abs(x)`.

See the [Quickstart](quickstart.qmd) for a complete model example.

## Panel preparation

Panel preparation batches forecast-row construction and reuses observation
indices when arranging group arrays, avoiding a group-by-observation selection
matrix. Formulas that only select existing columns also bypass expression
evaluation. Prediction rows and lag history are still prepared for both static
and dynamic models; missing-value filtering and unbalanced-panel padding are
unchanged.

Date-effect means reuse date-sorted observation indices built during panel
preparation, for both balanced and unbalanced panels and derivative arrays.
Final effect reporting reuses the fitted likelihood's variance components
and effect adjustments. Inference computes the sandwich covariance once when
the primary and fallback parameter sets coincide; constrained fits retain
separate calculations where required. These changes avoid repeated work
without changing the estimator, diagnostics or convergence criteria.

## Local single-start estimation

Fitting is local: `execute` prepares the panel once. Ordinary static FE/RE
models get fast FE/RE starting estimates from `processing/initreg.py` during
panel initialization, alongside the ARMA/GARCH starting-value selection.
All models pass their initialized parameter vector to `dfpmax` and use the
same likelihood optimizer and reporting. An initial convergence check lets
OLS and FE fits finish without a line search when the start already solves
the likelihood. RE starts are refined when necessary. Legacy
multiprocessing and GUI/browser progress channels have been removed, including
the `Model(..., multiprocess=...)` argument. Final summaries, standard errors,
diagnostics and predictions are retained and final statistics are computed once.

Multistart has also been removed: `OptimizerOptions(multistart=...)` is no
longer accepted, and results no longer expose `multistart_results`. Automatic
ARMA/GARCH initialization and explicit engine starting arguments are retained.
A single start does not guarantee a global likelihood maximum.
See [fit options](qmd/api/fit-options.qmd) for details and the Gaussian
random-effects likelihood audit.

Static and optimized fits use the same likelihood-based reporting, including
reported group/time effects. Two-way fixed effects use a joint projection
on unbalanced panels, rather than a single pass of separate demeaning.
One-way random-effect variance calculations account for observation counts
and only the requested effect dimension.

Random effects retain the residual-adaptive quasi-likelihood approximation:
ARMA innovations are adjusted for effects before the GARCH variance recursion.
This is not an exact likelihood obtained by integrating Gaussian random
effects. Static RE feasible-GLS estimates are starting values, not final
estimates: the shared optimizer refines them against the adaptive objective.
Explicit engine starting arguments are preserved. No separate static
inference convention or exact-integration engine is introduced.

## Errors and warnings

Unexpected errors in initialization, estimation and reporting propagate to
the caller rather than producing partial summaries or silently dropping
constraints. Common input mistakes retain contextual validation messages.
Fitting does not override Python warning filters; callers can choose whether
warnings are displayed, ignored or raised.

Numerical recovery is limited to intentional estimation behavior, such as
rejecting undefined likelihood trials and using a pseudo-inverse for a
singular optimizer Hessian. Initialization convergence warnings and warnings
about unavailable inference remain visible.

## Unavailable standard errors

Parameters with an entirely zero Hessian row and column have no local
curvature information. Paneltime preserves that zero curvature and reports
their standard errors and covariance rows/columns as `NaN`, with a warning,
while retaining inference for the remaining invertible block. Numerical
stabilization of the optimizer must not be interpreted as information about
such a coefficient.

## Pre-sample variance

ARMA innovations start without pre-sample residuals. Pre-sample conditional
variances and squared innovations use the residual variance of the initial
regression, held fixed throughout optimization. Every observed period,
including `t=0`, has the variance implied by the variance equation; no
observations are dropped for a variance burn-in. In particular,
`var_0 = omega_0 + init_var * (sum(gamma) + sum(psi))`.
With all ARCH and GARCH coefficients zero, the initial variance has no effect.
There is no estimated initial-variance parameter or derivative with respect
to it; derivatives with respect to the variance coefficients include their
pre-sample contributions.

The unconditional variance `omega / (1 - sum(gamma) - sum(psi))` is not used;
it depends on the estimated coefficients and is undefined for
`sum(gamma) + sum(psi) >= 1`. A fixed pre-sample start does not itself resolve
weak identification of individual GARCH coefficients.

## AR/MA common factors

With `lambda = -rho` the AR and MA polynomials share a factor that cancels,
and the likelihood is flat along that line, so ARMA terms fitted to data
without such dynamics drift along a ridge or stop at an arbitrary point on
it. All ARMA(1,1) impulse weights are proportional to `rho + lambda`. At
convergence paneltime tests the resultant of the AR and MA polynomials,
which is zero when they share a root (for ARMA(1,1), this reduces to
`rho + lambda`). Its delta-method standard error uses the free
mean-equation information block (beta, rho, lambda). If the absolute Wald
statistic is below 3 (a heuristic threshold, not a standard-normal test
under the unidentified null), the highest free AR and MA lags are fixed at
zero, turning ARMA(p,q) into ARMA(p-1,q-1), and the optimization continues.
The convergence message reports the removed lags.