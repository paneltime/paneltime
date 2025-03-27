---
title: Home
nav_order: 1
has_toc: true
---
# About

**Paneltime** is a statistical tool for estimating regressions on datasets that:  

- Are **panels** (have both a time and a group dimension)  
- Are **non-stationary in means** (ARIMA)  
- Are **non-stationary in variance** (GARCH)  

Unlike any other statistical tool currently available, **Paneltime** simultaneously estimates **random/fixed effects**, **ARIMA**, and **GARCH** parameters.  

The package can also be used on **non-panel data** or datasets that only exhibit ARIMA or GARCH characteristics. However, if your data has none of these issues, **OLS is the preferred method**.  

**Author:** Espen Sirnes  
**Current version:** 1.2.61  


# Installation


"pip install paneltime" for installation


# Usage

Datasets are estimated with 

```
paneltime.execute(model_string, dataframe, T,
						ID=None, HF=None,
						instruments=None, console_output=True)
```
It takes the following arguments:

- `model_string`: A string on the form 'Y ~ X1 + X2 + X3', where Y is the dependent and X1-X3 are the independents, as named in the dataframe (required)
- `dataframe`: a dataframe consisting of variables with the names used in `model_string` (required)
- `T`: the time identifier (required)
- `ID`: The group identifier (optional)
- `HF`: list with names of heteroskedasticity factors (optional)
- `instruments`: list with names of instruments (optional)
  
## The model string

The model string can contain operations supported by the `numpy` package, using `np` as alias for numpy. For example `np.abs(x)` will result in a 
variable that is the absolute value of x. 


# Example using world bank data
```
import wbdata
import paneltime as pt

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

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# Run the regression:
pt.options.pqdkm = (1, 1, 0, 1, 1)
m = pt.execute('Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))'
					 , df, timevar = 'date',idvar='country' )

# display results
print(m)


```





