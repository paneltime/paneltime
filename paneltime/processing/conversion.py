import numpy as np


def eval_variables(matrix, names,idvar_num):
	#this function is to be removed when it has been implemented in the panel module

	lag_obj = LagObject(pd_panel)
	d={'D':lag_obj.diff,'L':lag_obj.lag,'np':np}
	for i in df.keys():#Adding columns to name space
		d[i]=df[i]
	for i in x:
		if not i in df:
			try:
				df[i]=eval(i,d)
			except NameError as e:
				raise NameError(f"{i} not defined in data frame or function")

	n=len(df)
	df=df.dropna()
	lost_na_obs=(n-len(df))-lag_obj.max_lags

	return df, lost_na_obs, lag_obj.max_lags, n


class LagObject:
	#this class is to be removed when it has been implemented in the panel module
	def __init__(self,panel):
		self.panel=panel
		self.max_lags=0

	def lag(self,variable,lags=1):
		x=self.panel[variable.name].shift(lags)
		self.max_lags=max((self.max_lags,lags))
		return x

	def diff(self,variable,lags=1):
		x=self.panel[variable.name].diff(lags)
		self.max_lags=max((self.max_lags,lags))
		return x