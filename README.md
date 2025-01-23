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
