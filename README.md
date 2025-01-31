Author: Espen Sirnes
Version: 1.2.53

This package integrates paneldata estimation with a very efficient ARIMA/GARCH
estimator. 


# Installation


Use "pip install paneltime" for installation


# Usage

Datasets are estimated with 

```
paneltime.execute(model_string, dataframe, 
						ID=None,T=None,HF=None,
						instruments=None, console_output=True)
```
It takes the following arguments:

	- `model_string`: A string on the form 'Y ~ X1 + X2 + X3', where Y is the dependent and X1-X3 are the independents, as named in the dataframe.
	- `dataframe`: a dataframe consisting of variables with the names used in `model_string`.
	- `ID`: The group identifier
	- `T`: the time identifier
	- `HF`: list with names of heteroskedasticity factors
	- `instruments`: list with names of instruments
  

# Example using world bank data
```
import wbdata
import pandas as pd
import paneltime 

# Define variables to download
indicators = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_growth',    # BNP-vekst
    'FP.CPI.TOTL.ZG': 'Inflation',        # Inflasjon (konsumprisindeks)
    'FR.INR.LEND': 'Interest_rate',        # Rentenivå (lånerente)
	'NY.GNS.ICTR.ZS': 'Gross_Savings',  # Gross savings (% of GDP)
    'NE.CON.GOVT.ZS': 'Gov_Consumption',  # Government consumption (% of GDP)
}


# Download data
df = wbdata.get_dataframe(indicators)


# prepare data
df = pd.DataFrame(df.reset_index())
df = df.rename(columns = {'date':'year'})
df = df.sort_values(by=['country', 'year'])
df_grouped = df.groupby('country')


df['Lagged_Gross_Savings'] = df_grouped['Gross_Savings'].shift(1)
df['Lagged_Gov_Consumption'] = df_grouped['Gov_Consumption'].shift(1)
df['Lagged_Growth'] =          df_grouped['GDP_growth'].shift(1)
df['Lagged_Inflation'] =       df_grouped['Inflation'].shift(1)
df['Lagged_Interest_rate'] =   df_grouped['Interest_rate'].shift(1)

df = df[abs(df['Inflation'])<30]



# Estimate:
m = paneltime.execute('Inflation~Intercept+Lagged_Growth+Lagged_Inflation+Lagged_Interest_rate+'
					  'Lagged_Gross_Savings', df,T = 'year',ID='country' )

# display results
print(m)

```


# Setting options

You can set various options by setting attributes of the `options` attribute, for example: 
```
import paneltime as pt
pt.options.add_intercept = False
```

The options available are:

{% paneltime/options.md %}


# Ouput

`paneltime.execute()` returns a `Summary`-object. This object has the following attributes and methods:

## `Summary` attributes
| Attribute        | Sub attributes               | Data Type           | Explanation                                                                                          |
|------------------|------------------------------|---------------------|------------------------------------------------------------------------------------------------------|
| `count`         |                              | `Counting` instance | Contains counts of the sample                                                                        |
|                | `count_dates`                | `float`             | The number of time observations in the sample after filtering                                        |
|                | `count_deg_freedom`          | `float`             | Degrees of freedom                                                                                   |
|                | `count_groups`               | `float`             | The number of groups in the sample after filtering                                                   |
|                | `count_samp_size_after_filter` | `float`           | The number of observations after filtering                                                            |
|                | `count_samp_size_orig`       | `float`             | The number of original observations                                                                   |
| `general`      |                              | `General` instance  | Contains various information of interest                                                              |
|                | `CI`                         | `float`             | The condition index                                                                                  |
|                | `CI_n`                       | `int`               | The number of variables being dependent (more than 50% of variance) of a high CI>30 factor            |
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


