Author: Espen Sirnes
Version: 1.2.50

This package integrates paneldata estimation with a very efficient ARIMA/GARCH
estimator. 


# Installation


Use "pip install paneltime" for installation


# Usage

Datasets are estimated with 

```
paneltime.execute(model_string, dataframe, ID=None,T=None,HF=None,instruments=None, console_output=True)
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


|Attribute name|Default<br>value|Permissible<br>values|Data<br>type|Description|
|--------------|-------------|-----------|-----------|-----------|
|ARMA_constraint|1000|None|float|<b>ARMA coefficient constraint:</b> Maximum absolute value of ARMA coefficients|
|ARMA_round|14|%s>0|int|<b># of signficant digits:</b> Number og digits to round elements in the ARMA matrices by. Small differences in these values can change the optimization path and makes the estimate less robustNumber of significant digits in ARMA|
|EGARCH|False|[True, False]|bool|<b>Estimate GARCH directly:</b> Normal GARCH, as opposed to EGARCH if True|
|GARCH_assist|0|%s>=0|float|<b>GARCH assist:</b> Amount of weight put on assisting GARCH variance to be close to squared residuals|
|accuracy|0|%s>0|int|<b>Accuracy:</b> Accuracy of the optimization algorithm. 0 = fast and inaccurate, 3=slow and maximum accuracy|
|add_intercept|True|[True, False]|bool|<b>Add intercept:</b> If True, adds intercept if not all ready in the data|
|arguments|None|None|['str', 'dict', 'list', 'ndarray']|<b>Initial arguments:</b> A dict or string defining a dictionary in python syntax containing the initial arguments.An example can be obtained by printing ll.args.args_d|
|constraints_engine|True|[True, False]|bool|<b>Uses constraints engine:</b> Determines whether to use the constraints engine|
|fixed_random_group_eff|0|[0, 1, 2]|int|<b>Group fixed random effect:</b> Fixed, random or no group effects|
|fixed_random_time_eff|0|[0, 1, 2]|int|<b>Time fixed random effect:</b> Fixed, random or no time effects|
|fixed_random_variance_eff|0|[0, 1, 2]|int|<b>Variance fixed random effects:</b> Fixed, random or no group effects for variance|
|h_function|def h(e,z...|None|str|<b>GARCH function:</b> You can supply your own heteroskedasticity function. It must be a function of<br>residuals e and a shift parameter z that is determined by the maximization procedure<br>the function must return the value and its computation in the following order:<br>h, dh/de, (d^2)h/de^2, dh/dz, (d^2)h/dz^2,(d^2)h/(dz*de)|
|include_initvar|True|[True, False]|bool|<b>Include initial variance:</b> If True, includes an initaial variance term|
|initial_arima_garch_params|0.1|%s>=0|float|<b>initial size of arima-garch parameters:</b> The initial size of arima-garch parameters (all directions will be attempted|
|kurtosis_adj|0|%s>=0|float|<b>Amount of kurtosis adjustment:</b> Amount of kurtosis adjustment|
|max_iterations|150|%s>0|int|<b>Maximum number of iterations:</b> Maximum number of iterations|
|min_group_df|1|%s>0|int|<b>Minimum degrees of freedom:</b> The smallest permissible number of observations in each group. Must be at least 1|
|multicoll_threshold_max|1000|None|float|<b>Multicollinearity threshold:</b> Threshold for imposing constraints on collineary variables|
|multicoll_threshold_report|30|None|float|<b>Multicollinearity threshold:</b> Threshold for reporting multicoll problems|
|pqdkm|[1, 1, 0, 1, 1]|%s>=0|int|<b>ARIMA-GARCH orders:</b> ARIMA-GARCH parameters:|
|robustcov_lags_statistics|[100, 30]|%s>1|int|<b>Robust covariance lags (time):</b> Numer of lags used in calculation of the robust <br>covariance matrix for the time dimension|
|subtract_means|False|[True, False]|bool|<b>Subtract means:</b> If True, subtracts the mean of all variables. This may be a remedy for multicollinearity if the mean is not of interest.|
|supress_output|True|[True, False]|bool|<b>Supress output:</b> If True, no output is printed.|
|tobit_limits|[None, None]|None|['float', 'NoneType']|<b>Tobit-model limits:</b> Determines the limits in a tobit regression. Element 0 is lower limit and element1 is upper limit. If None, the limit is not active|
|tolerance|0.001|%s>0|float|<b>Tolerance:</b> Tolerance. When the maximum absolute value of the gradient divided by the hessian diagonalis smaller than the tolerance, the procedure is Tolerance in maximum likelihood|
|use_analytical|1|[0, 1, 2]|int|<b>Analytical Hessian:</b> Use analytical Hessian|
|user_constraints|None|None|['str', 'dict']|<b>User constraints:</b> You can add constraints as a dict or as a string in python dictonary syntax.<br>|
|variance_RE_norm|5e-06|%s>0|float|<b>Variance RE/FE normalization point in log function:</b> This parameter determines at which point the log function involved in the variance RE/FE calculations, will be extrapolate by a linear function for smaller values|
