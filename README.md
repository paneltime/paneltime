
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

## Unavailable standard errors

Parameters with an entirely zero Hessian row and column have no local
curvature information. Paneltime preserves that zero curvature and reports
their standard errors and covariance rows/columns as `NaN`, with a warning,
while retaining inference for the remaining invertible block. For example,
a variance factor active only in the first period has no effect when that
period's variance is fixed by initialization. Numerical stabilization of the
optimizer must not be interpreted as information about such a coefficient.