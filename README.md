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

{% paneltime/options.md %}
