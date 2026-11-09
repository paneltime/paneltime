
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

**Current version:** 1.3.8

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

## Model documentation

The full model documentation is available as a
[PDF](https://paneltime.github.io/paneltime_documentation.pdf).

## ARFIMA

Pass `fractional_d="estimate"` to estimate a common stationary fractional
parameter, or a fixed value such as `fractional_d=0.25`.
Use `order=(p, q, 0)` (AR, MA, zero integer differencing).
The conditional filter uses zero pre-sample residuals and requires regular
observations without internal gaps. It works with the existing innovation
likelihoods and GARCH models.
See [fit options](api/fit-options.qmd#fractional-differencing-arfima) for details.

## ARIMA initial-state likelihood

The default `arima_initialization="conditional"` retains zero presample ARMA
states and existing behavior. Set `arima_initialization="stationary"` in `fit()`
to integrate a stationary Gaussian presample ARMA state independently per entity.
With constant innovation variance and no effects, this gives exact Gaussian
ARMA likelihood after integer differencing, conditional on the initial levels.
The score, Hessian, per-observation scores, standard errors, standardized
residuals and one-step forecasts use the same integrated likelihood.

With GARCH or panel effects, their existing recursion/transformation is retained.
Only the ARMA initial state is integrated, with its unit-noise stationary
covariance scaled by each entity's first conditional innovation variance.
This is **not** exact joint GARCH or marginal random-effects likelihood.
It requires regular observations, stationary AR roots, a built-in Gaussian
likelihood and no fractional differencing or censoring. Dense covariance
factorization costs cubic time and quadratic storage in each history length;
benchmark long series before large simulations.
See [fit options](api/fit-options.qmd#arima-initial-state-likelihood) for the formula.

## GARCH ridge reduction

`pt.OptimizerOptions(garch_ridge_wald=0.2)` controls automatic reduction of
weakly identified GARCH lag splits. The default `0.2` is a squared Wald
threshold, not a continuous ridge penalty. Use `0` to disable reduction.
Larger thresholds make reduction attempts more likely, but the refit must
converge and pass a likelihood-ratio test before it is retained.
See [fit options](api/fit-options.qmd#garch-ridge-reduction) for details.