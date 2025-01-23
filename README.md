# Paneltime


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

# Estimate:
m = paneltime.execute('Inflation~Intercept+Lagged_Growth+Lagged_Inflation+Lagged_Interest_rate+'
					  'Lagged_Gross_Savings+Diff_Gov_Consumption', pt_data,T = 'Year',ID='Country' )

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


|Attribute name|Default value|Description|
|--------------|-------------|-----------|
|accuracy|0|int|Accuracy:Accuracy of the optimization algorithm. 0 = fast and inaccurate, 3=slow and maximum accuracy|
|add_intercept|True|bool|Add intercept:If True, adds intercept if not all ready in the data|
|arguments|None|['str', 'dict', 'list', 'ndarray']|Initial arguments:A dict or string defining a dictionary in python syntax containing the initial arguments.An example can be obtained by printing ll.args.args_d|
|ARMA_constraint|1000|float|ARMA coefficient constraint:Maximum absolute value of ARMA coefficients|
|betaconstraint|True|bool|Constraint betas initially:Determines whether to initially constraint beta coefficients while setting the ARIMA-GARCH-coefficients|
|constraints_engine|True|bool|Uses constraints engine:Determines whether to use the constraints engine|
|debug_mode|False|bool|Debug or not:Determines whether the code will run in debug mode. Should normally allways be False|
|multicoll_threshold_report|30|float|Multicollinearity threshold:Threshold for reporting multicoll problems|
|multicoll_threshold_max|1000|float|Multicollinearity threshold:Threshold for imposing constraints on collineary variables|
|EGARCH|False|bool|Estimate GARCH directly:Normal GARCH, as opposed to EGARCH if True|
|do_not_constrain|None|['str', 'NoneType']|Avoid constraint:The name of a variable of interest that shall not be constrained due to multicollinearity|
|fixed_random_group_eff|0|int|Group fixed random effect:Fixed, random or no group effects|
|fixed_random_time_eff|0|int|Time fixed random effect:Fixed, random or no time effects|
|fixed_random_variance_eff|0|int|Variance fixed random effects:Fixed, random or no group effects for variance|
|h_function|def h(e,z):<function definition>|str|GARCH function:You can supply your own heteroskedasticity function. It must be a function ofresiduals e and a shift parameter z that is determined by the maximization procedurethe function must return the value and its computation in the following order:h, dh/de, (d^2)h/de^2, dh/dz, (d^2)h/dz^2,(d^2)h/(dz*de)|
|include_initvar|True|bool|Include initial variance:If True, includes an initaial variance term|
|initial_arima_garch_params|0.1|float|initial size of arima-garch parameters:The initial size of arima-garch parameters (all directions will be attempted|
|kurtosis_adj|0|float|Amount of kurtosis adjustment:Amount of kurtosis adjustment|
|GARCH_assist|0|float|GARCH assist:Amount of weight put on assisting GARCH variance to be close to squared residuals|
|min_group_df|1|int|Minimum degrees of freedom:The smallest permissible number of observations in each group. Must be at least 1|
|max_iterations|150|int|Maximum number of iterations:Maximum number of iterations|
|max_increments|0|float|Maximum increments:Maximum increment before maximization is ended|
|minimum_iterations|0|int|Minimum iterations:Minimum number of iterations in maximization:|
|pool|False|bool|Pooling:True if sample is to be pooled, otherwise False.For running a pooled regression|
|pqdkm|[1, 1, 0, 1, 1]|int|ARIMA-GARCH orders:ARIMA-GARCH parameters:|
|robustcov_lags_statistics|[100, 30]|int|Robust covariance lags (time):Numer of lags used in calculation of the robust covariance matrix for the time dimension|
|silent|False|bool|Silent mode:True if silent mode, otherwise False.For running the procedure in a script, where output should be suppressed|
|subtract_means|False|bool|Subtract means:If True, subtracts the mean of all variables. This may be a remedy for multicollinearity if the mean is not of interest.|
|supress_output|True|bool|Supress output:If True, no output is printed.|
|tobit_limits|[None, None]|['float', 'NoneType']|Tobit-model limits:Determines the limits in a tobit regression. Element 0 is lower limit and element1 is upper limit. If None, the limit is not active|
|tolerance|0.001|float|Tolerance:Tolerance. When the maximum absolute value of the gradient divided by the hessian diagonalis smaller than the tolerance, the procedure is Tolerance in maximum likelihood|
|ARMA_round|14|int|# of signficant digits:Number og digits to round elements in the ARMA matrices by. Small differences in these values can change the optimization path and makes the estimate less robustNumber of significant digits in ARMA|
|variance_RE_norm|5e-06|float|Variance RE/FE normalization point in log function:This parameter determines at which point the log function involved in the variance RE/FE calculations, will be extrapolate by a linear function for smaller values|
|user_constraints|None|['str', 'dict']|User constraints:You can add constraints as a dict or as a string in python dictonary syntax.|
|use_analytical|1|int|Analytical Hessian:Use analytical Hessian|
|web_open_tab|True|bool|New web tab:True if web a new web browser tab should be opened when using web interfaceShould a new tab be opemed?|
