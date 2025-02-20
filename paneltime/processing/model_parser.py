#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEFAULT_INTERCEPT_NAME='Intercept'
VAR_INTERCEPT_NAME='log_variance_constant'
INSTRUMENT_INTERCEPT_NAME='instrument_intercept'
CONST_NAME='one'
NUMERIC_TAG="_numeric"

import numpy as np
import pandas as pd
import builtins
import keyword

def get_variables(ip, df,model_string,idvar,timevar,heteroscedasticity_factors,instruments,settings,pool=(None,'mean')):
	if not settings.supress_output:
		print ("Analyzing variables ...")
	if not type(df)==pd.DataFrame:
		raise RuntimeError('The dataframe supplied is not a pandas dataframe. Only pandas dataframes are supported.')
	if CONST_NAME in df:
		print(f"Warning: The name {CONST_NAME} is reserved for the constant 1."
											f"The variable with this name will be overwritten and set to 1")
	
	df = df.reset_index()
	
	df=pool_func(df,pool)

	df[CONST_NAME]=1
	df[DEFAULT_INTERCEPT_NAME]     = df[CONST_NAME]
	df[VAR_INTERCEPT_NAME]         = df[CONST_NAME]
	df[INSTRUMENT_INTERCEPT_NAME]  = df[CONST_NAME]

	identify_sort_var(timevar, df)
	identify_sort_var(idvar, df)
	#if not pd.api.types.is_datetime64_any_dtype(df['date_column']):

	idvar   = get_names(idvar, df,  'id variable')
	timevar = get_names(timevar, df,  'time variable')

	idvar_num,     idvar      = handle_idvar(ip,df,idvar)
	timevar_num, timevar  = handle_time(ip,df,timevar)
	sort= idvar_num + timevar
	if len(sort):
		df=df.sort_values(sort)

	W=get_names(heteroscedasticity_factors, df,'heteroscedasticity_factors',True, VAR_INTERCEPT_NAME)
	Z=get_names(instruments, df,'instruments',True,INSTRUMENT_INTERCEPT_NAME)

	try:
		Y,X=parse_model(model_string, settings)
	except:
		raise RuntimeError("The model_string must be on the form Y~X1+X2+X3")
	if Y==['']:
		raise RuntimeError("No dependent variable specified")

	x = idvar_num+idvar+timevar+timevar_num+W+Z+Y+X
	x = list(dict.fromkeys(x))
	df, ip.lost_na_obs, ip.max_lags, ip.orig_n = eval_variables(df, x, idvar_num)

	if len(df)==0:
		raise RuntimeError('The filtered data is. This typically happens if all observations have nan-observations. Plealse check your data.')
	const={}
	for x,add_intercept,num in [
					('idvar_num',False,True),('timevar_num',False,True),
																('idvar',False,False),('timevar',False,False),
																('W',True,True),('Z',True,True),('Y',False,True),
																	('X',settings.add_intercept,True)]:
		ip.__dict__[x], const[x]= check_var(df,locals()[x],x,add_intercept,num)
		if ip.__dict__[x] is None:
			ip.__dict__[x + "_names"] = None
		else:
			ip.__dict__[x + "_names"] = list(ip.__dict__[x].columns)
	ip.dataframe=df
	ip.has_intercept=const['X']

def identify_sort_var(x, df):
	if x is None:
		return
	if x in df:
		return
	if x == df.index.name:
		df[x] = df.index
	elif x in df.index.names:
		df[x] = df.index[x]
	else:
		raise KeyError(f"Name {x} not found in data frame")

def pool_func(df,pool):
	x,operation=pool
	if x is None:
		return df
	x=get_names(x, 'pool')
	df=df.groupy(x).agg(operation)
	return df

def check_var(df,x,inputtype,add_intercept,numeric):
	if len(x)==0:
		return None,None
	try:
		dfx=df[x]
	except KeyError as e:
		x = [i.replace(' ', '_') for i in x]
		dfx=df[x]
	if not numeric:
		return dfx,None
	const_found=False
	for i in x:
		if ' ' in i:
			raise RuntimeError(f'Spaces are not allowed in variables, but found in the variable {i} from {inputtype}')
		try:
			v=np.var(dfx[i])
		except TypeError as e:
			raise TypeError(f"All variables except time and id must be numeric. {e}")
		if v==0 and const_found:
			if dfx[i].iloc[0]==0:
				print(f"Warning: All values in {i} from {inputtype} are zero, variable dropped")
			else:
				print(f"Warning: {i} from {inputtype} is constant. Variable dropped.")
			dfx=dfx.drop(i,1)
		elif v==0 and not const_found:
			if inputtype=='Y':
				raise RuntimeError('The dependent variable is constant')
			const_found=True
	return dfx,const_found

def eval_variables(df, x,idvar_num):
	pd_panel = df
	if len(idvar_num)>0:
		pd_panel=df.groupby(idvar_num)
	lag_obj=lag_object(pd_panel)
	d={'D':lag_obj.diff,'L':lag_obj.lag,'np':np}
	for i in df.keys():#Adding columns to name space
		d[i]=df[i]
	for i in x:
		if not i in df:
			try:
				df[i]=eval(i,d)
			except:
				raise NameError(f"{i} not defined in data frame or function")
	try:#just to make sure everytning is ok
		df = pd.DataFrame(df[x])
	except KeyError:
		df_test(x, df)

	n=len(df)
	df=df.dropna()
	lost_na_obs=(n-len(df))-lag_obj.max_lags

	return df, lost_na_obs, lag_obj.max_lags, n

def df_test(x, df):
	not_in = []
	for i in x:
		if not i in df:
			not_in.append(i)
	raise RuntimeError(f"These names are in the model, but not in the data frame:{', '.join(not_in) }")


class lag_object:
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

def parse_model(model_string,settings):
	split = None
	for i in ['~','=']:
		if i in model_string:
			split=i
			break
	if split is None:#No dependent
		return [model_string],[DEFAULT_INTERCEPT_NAME]
	Y,X=model_string.split(split)
	X=[i.strip() for i in X.split('+')]
	Y = Y.strip()
	if X==['']:
		X=[DEFAULT_INTERCEPT_NAME]
	if settings.add_intercept and not (DEFAULT_INTERCEPT_NAME in X):
		X=[DEFAULT_INTERCEPT_NAME]+X
	return [Y], ordered_unique(X)


def ordered_unique(X):
	unique = []
	invalid = ['']
	for i in X:
		if not i in unique + invalid:
			unique.append(i)
	return unique


def get_names(x, df,inputtype,add_intercept=False,intercept_name=None):
	r = None
	if x is None:
		r=[]
	elif type(x)==str:
		r=[x]
	elif type(x)==list or type(x)==tuple:
		r=list(x.name)
	
	if r is None or not np.all(i in df for i in r):
		raise RuntimeError(f"Input for {inputtype} needs to be a string, list or tuple of strings," 
					 		"corresponding to names in the supplied data frame")
	
	if add_intercept:
		r=[intercept_name]+r

	return list(np.unique(r))

def handle_time(ip,df,x):
	if x==[]:
		return [],[]
	x=x[0]

	if np.issubdtype(np.array(df[x]).dtype, np.number):
		df[x+NUMERIC_TAG] = df[x]
		return [x+NUMERIC_TAG],[x]

	try:
		x_dt=pd.to_datetime(df[x])
	except ValueError as e:
		try:
			x_dt=pd.to_numeric(x_dt)
		except ValueError as e:
			raise ValueError(f'Expected date or numeric for {inputtype}, but {x} is not recognized as a date or numeric variable by pandas.')
	x_dt=pd.to_numeric(x_dt)/(24*60*60*1000000000)
	if np.all(x_dt.astype(int)==x_dt):
		x_dt=x_dt.astype(int)
	df[x+NUMERIC_TAG]=x_dt
	return [x+NUMERIC_TAG],[x]


def handle_idvar(ip,df,x):
	if x==[]:
		return [],[]
	x=x[0]
	ids, ip.idvar_unique = pd.factorize(df[x],True)
	df[x+NUMERIC_TAG]=ids
	#both these are true before next assignment:
	#np.all(ip.idvar_unique[ids]==df[x])
	#np.all(np.arange(len(ids))[ids]==ids)
	return [x+NUMERIC_TAG],[x]
