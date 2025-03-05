---
title: Output
nav_order: 3
has_toc: true
---
# Output


`paneltime.execute()` returns a `Summary`-object. In addition, printing the `Summary` object, prints out the regression table and related information.


This object has the following attributes:

## Attributes available in the `Summary` object


| Attribute        | Sub attributes               | Data Type           | Explanation                                                                                          |
|------------------|------------------------------|---------------------|------------------------------------------------------------------------------------------------------|
| `count`         |                              | `Counting` instance | Contains counts of the sample                                                                        |
|                | `count_dates`                | `float`             | The number of time observations in the sample after filtering                                        |
|                | `count_deg_freedom`          | `float`             | Degrees of freedom                                                                                   |
|                | `count_groups`               | `float`             | The number of groups in the sample after filtering                                                   |
|                | `count_samp_size_after_filter` | `float`           | The number of observations after filtering                                                            |
|                | `count_samp_size_orig`       | `float`             | The number of original observations                                                                   |
| `general`      |                              | `General` instance  | Contains various information of interest                                                              |
|                | `ci`                         | `float`             | The condition index                                                                                  |
|                | `ci_n`                       | `int`               | The number of variables being dependent (more than 50% of variance) of a high ci>30 factor            |
|                | `converged`                  | `float`             | Indicates whether the maximization procedure converged (`True`) or not (`False`)                     |
|                | `dx_norm`                    | `float`             | The normalized direction for the next iteration. Should be close to 0 for all variables.              |
|                | `gradient_matrix`            | `float`             | The gradient matrix of vectors for each observation, calculated analytically at the final regression coefficients point. |
|                | `gradient_vector`            | `float`             | The gradient vector, calculated analytically at the final regression coefficients point.               |
|                | `hessian`                    | `float`             | The Hessian matrix, calculated analytically at the point of the final regression coefficients point.    |
|                | `its`                        | `int`               | Number of iterations used before the maximum was identified                                         |
|                | `log_likelihood`             | `float`             | The log-likelihood of the regression                                                                   |
|                | `log_likelihood_object`      | `LL` instance       | An object containing the log-likelihood function and data needed to calculate the gradient and Hessian|
|                | `msg`                        | `float`             | Message from the maximization procedure                                                                |
|                | `panel`                      | `float`             | An object containing the data and information about the panel                                           |
|                | `t0`                         | `float`             | Start time for the maximization                                                                        |
|                | `t1`                         | `float`             | End time for the maximization                                                                          |
|                | `time`                       | `float`             | The time the regression took                                                                            |
| `names`        |                              | `Names` instance    | Contains names of variables in various formats                                                         |
|                | `captions`                   | `list`              | List containing captions for each variable, as displayed in the output table                            |
|                | `groups`                     | `list`              | List containing the names of each variable group. For example, 'beta' is the regression coefficient group and 'lambda' is the MA-coefficients group |
|                | `varnames`                   | `list`              | List containing the internal variable names used in the code                                            |
| `output`       |                              | `Output` instance   | For internal use                                                                                       |
| `random_effects` |                            | `RandomEffects` instance | Estimated random/fixed effects                                                                         |
|                | `residuals_i`                | `numpy.ndarray`     | Estimated random/fixed effects for the group dimensions                                                |
|                | `residuals_t`                | `numpy.ndarray`     | Estimated random/fixed effects for the time dimensions                                                 |
|                | `std_i`                      | `numpy.ndarray`     | Estimated volatilities for the group dimensions                                                        |
|                | `std_t`                      | `numpy.ndarray`     | Estimated volatilities for the time dimensions                                                         |
| `results`      |                              | `Results` instance  | Contains regression statistics seen in the regression output table                                      |
|                | `args`                       | `dict`              | Dictionary with the variable groups in `results.names.groups` as keys, containing nx1 matrices of parameter estimates |
|                | `codes`                      | `float`             | Significance codes as displayed in the table                                                            |
|                | `conf_025`                   | `float`             | Lower 5% confidence interval                                                                             |
|                | `conf_0975`                  | `float`             | Upper 5% confidence interval                                                                             |
|                | `params`                     | `float`             | List of regression parameter estimates in the same order as `results.names.captions`                    |
|                | `se`                         | `float`             | Standard errors                                                                                         |
|                | `tstat`                      | `float`             | T-statistics                                                                                            |
|                | `tsign`                      | `float`             | T-test significance levels                                                                              |
| `table`        |                              | `RegTableObj` instance | An object that generates tables. Mostly for internal use.                                              |


