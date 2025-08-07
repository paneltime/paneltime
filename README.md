


**Paneltime** is a statistical tool for estimating regressions on datasets that:  

- Are **panels** (have both a time and a group dimension)  
- Are **non-stationary in means** (ARIMA)  
- Are **non-stationary in variance** (GARCH)  

Unlike any other statistical tool currently available, **Paneltime** simultaneously handles common timeseries problems 
(**ARIMA**/**GARCH**)  and  *panel data sets* consisting of several groups/individuals 
 (**random/fixed effects**).  

The package can also be used on pure **panel data** sets without ARIMA/GARCH or single time series data with ARIMA/GARCH problems. 

**Author:** Espen Sirnes  
**Current version:** 1.2.68  


# Installation


"pip install paneltime" for installation


# Usage

Datasets are estimated with 

```
paneltime.execute(model_string, dataframe, T,
						ID=None, HF=None,
						instruments=None)
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
```python
{{< include example.py >}}
```

[Download working example zip](working_example.zip)



