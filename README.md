
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

**Current version:** 1.3.9

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

## Simulations and testing

The simulation code, parameter settings, benchmark runners, and analysis scripts
are available in the [testing repository](https://github.com/paneltime/paneltime_test).
The "Simulation procedure" section of the model documentation describes the
data-generating design and assumed parameters.

## ARFIMA

Pass `fractional_d="estimate"` to estimate a common stationary fractional
parameter, or a fixed value such as `fractional_d=0.25`.
Use `order=(p, q, 0)` (AR, MA, zero integer differencing).
The conditional filter uses zero pre-sample residuals and requires regular
observations without internal gaps. It works with the existing innovation
likelihoods and GARCH models.
See [fit options](api/fit-options.qmd#fractional-differencing-arfima) for details.

## GARCH ridge reduction

`pt.OptimizerOptions(garch_ridge_wald=0.2)` controls automatic reduction of
weakly identified GARCH lag splits. The default `0.2` is a squared Wald
threshold, not a continuous ridge penalty. Use `0` to disable reduction.
Larger thresholds make reduction attempts more likely, but the refit must
converge and pass a likelihood-ratio test before it is retained.
See [fit options](api/fit-options.qmd#garch-ridge-reduction) for details.