#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEFAULT_INTERCEPT_NAME='Intercept'
VAR_INTERCEPT_NAME='log_variance_constant'
INSTRUMENT_INTERCEPT_NAME='instrument_intercept'
CONST_NAME='one'
ORIG_SUFIX = '_orig'


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
	
	
	timevar, idvar = check_dimensions(df, timevar, idvar)


	df = df.reset_index()

	if df.columns.duplicated().any():
		raise RuntimeError(f"There are more than one occurence of '{df.columns[df.columns.duplicated()][0]}'. " 
					 "All variable names must be unique")
	
	df=pool_func(df,pool)

	df[CONST_NAME]=1
	df[DEFAULT_INTERCEPT_NAME]     = df[CONST_NAME]
	df[VAR_INTERCEPT_NAME]         = df[CONST_NAME]
	df[INSTRUMENT_INTERCEPT_NAME]  = df[CONST_NAME]

	identify_sort_var(timevar, df)
	identify_sort_var(idvar, df)


	idvar   = get_names(idvar, df,  'id variable')
	timevar = get_names(timevar, df,  'time variable')




	sort= idvar + timevar
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

	vars = W+Z+Y+X

	idvar_orig = numberize_idvar(ip,df,timevar, idvar)
	timevar_orig, time_delta, time_delta_orig  = numberize_time(df,timevar, idvar)

	df, ip.lost_na_obs, ip.max_lags, ip.orig_n = eval_and_add_pred_space(df, vars, idvar_orig, timevar_orig, 
																	idvar, timevar, time_delta, time_delta_orig)



	df_test(vars, df)

	if len(df)==0:
		raise RuntimeError('The filtered data is. This typically happens if all observations have nan-observations. Plealse check your data.')
	
	ip.has_intercept = add_variables(ip, settings, df, locals(), idvar)
	ip.dataframe=df


def add_variables(ip, settings, df, locals, idvar):
	const={}
	df = df.reset_index()
	for x,add_intercept,num in [('idvar',False,True),('timevar',False,True),
							 	('idvar_orig',False,False),('timevar_orig',False,False),
								('W',True,True),('Z',True,True),('Y',False,True),
								('X',settings.add_intercept,True)]:
		
		var, const[x]= check_var(df,locals[x],x,num)
		
		if var is None:
			ip.__dict__[x + "_names"] = None
		else:
			ip.__dict__[x + "_names"] = list(var)


		ip.__dict__[x] = var

	return const['X']
		
		
def check_dimensions(df, timevar, idvar):
	ix = df.index

	if (not timevar is None) or isinstance(ix, pd.RangeIndex):
		return timevar, idvar
	
	for k in range(len(ix.names)):
		try:
			pd.to_datetime(ix.get_level_values(k), format='%Y-%m-%d')
			names = list(ix.names)
			timevar = names.pop(k)
			if idvar  is None:
				unique_dates = len(ix.get_level_values(timevar))
				unique_indicies = len(set(df.index.to_list()))
				if unique_dates != unique_indicies:
					idvar = names[0]
			return timevar, idvar
		except Exception as e:
			pass

	return timevar, idvar

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




def check_var(df,x,inputtype,numeric):
	if len(x)==0:
		return None,None
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


def eval_and_add_pred_space(df, vars, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig):
	df = df.sort_values(idvar_orig + timevar_orig)
	df = df.set_index(idvar_orig + timevar_orig)
	df_new, lost_na_obs, max_lags, n = eval_variables(df, idvar + timevar + vars, idvar_orig)
	if max_lags>0:
		df_new = extend_timeseries(df, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig, max_lags)
		df_new, lost_na_obs, max_lags, n = eval_variables(df_new, idvar + timevar + vars, idvar_orig)
	df_new=df_new.dropna()
	
	#todo: 
	# - add max_lags observations to the end of all variables that have dates at the terminal date
	# - cut the corresponding obsverations in the arrayized matrices in the panel module, and 
	#   assign them to prediction versions. 
	return df_new, lost_na_obs, max_lags, n


def extend_timeseries(df, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig, max_lags):
	# Ensure the data is sorted by timevar within each group

	return df

	#Works here to cut of last rows in `panel` before can be implemented.
	
	idvar_orig, timevar_orig = idvar_orig[0], timevar_orig[0]
	idvar, timevar = idvar[0], timevar[0]

	# Find the maximum date in the dataset
	max_date = df.index.get_level_values(timevar_orig).max()

	# Get the last date per group
	last_dates = df.groupby(level=idvar_orig).apply(lambda x: x.index.get_level_values(timevar_orig).max())
	last_dates = last_dates.reset_index()
	last_dates.columns = [idvar_orig, timevar_orig]  # Ensure proper column names

	# Filter groups where last date matches max_date
	extend_groups = last_dates[last_dates[timevar_orig] == max_date]

	# Compute mean of previous observations per group
	group_means = df.groupby(level=idvar_orig).mean()

	# Prepare new rows
	new_rows = []
	
	for _, row in extend_groups.iterrows():
		group_id = row[idvar_orig]
		last_date_orig = row[timevar_orig]
		last_date = df.loc[group_id,last_date_orig][timevar]
		mean_values = group_means.loc[group_id]  # Get mean values for the group

		for i in range(1, max_lags + 1):
			new_date_orig = last_date_orig + i * time_delta_orig  # Increment date
			new_row = pd.Series(mean_values, name=(group_id, new_date_orig))  # Set new index tuple
			new_row[timevar] = last_date + i * time_delta
			new_rows.append(new_row)
	
	# Convert new rows to DataFrame
	if new_rows:
		new_df = pd.DataFrame(new_rows)
		new_df.index.names = [idvar_orig, timevar_orig]  # Ensure multi-index naming
		df_extended = pd.concat([df, new_df]).sort_index()
	else:
		df_extended = df

	return df_extended



def eval_variables(df, x,idvar_orig):
	new_df = pd.DataFrame()
	pd_panel = df
	if len(idvar_orig)>0:
		pd_panel=df.groupby(level=idvar_orig)
	lag_obj = LagObject(pd_panel)

	n=len(df)
	df=df.dropna()
	lost_na_obs = (n-len(df))

	d={'D':lag_obj.diff,'L':lag_obj.lag,'np':np}
	for i in df.keys():#Adding columns to name space
		d[i]=df[i]
	for i in x:
		if not i in df:
			try:
				new_df[i] = eval(i,d)
			except NameError as e:
				raise NameError(f"{i} not defined in data frame or function")
		else:
			new_df[i] = df[i]
	
	if len(idvar_orig):
		maxlags = max(new_df.isna().groupby(level = idvar_orig).sum().max())
	else:
		maxlags = max(new_df.isna().sum().max())

	return new_df, lost_na_obs, maxlags, n


class LagObject:
	def __init__(self,panel):
		self.panel=panel

	def lag(self,variable,lags=1):
		x = variable.shift(lags)
		return x

	def diff(self,variable,lags=1):
		x = variable.diff(lags)
		return x

def df_test(x, df):
	try: 
		df = pd.DataFrame(df[x])
	except KeyError:
		not_in = []
		for i in x:
			if not i in df:
				not_in.append(i)
		raise RuntimeError(f"These names are in the model, but not in the data frame:{', '.join(not_in) }")


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

def numberize_time(df, timevar, idvar):
	if timevar==[]:
		return [],None, None
	timevar=timevar[0]
	timevar_orig = timevar+ ORIG_SUFIX

	#Trying to coerce to number
	try:
		df[timevar] = df[timevar].astype(float)
		df_int = df[timevar].astype(int)
		if np.all(df[timevar]==df_int):
			df[timevar] = df_int
	except:
		pass

	#if number:
	dtype = np.array(df[timevar]).dtype

	if np.issubdtype(dtype, np.number):
		df[timevar + ORIG_SUFIX] = df[timevar]
		time_delta = get_mean_diff(df, timevar, idvar)
		if np.issubdtype(dtype, np.integer):
			time_delta = int(time_delta)
			if time_delta == 0:
				time_delta = 1
		return [timevar_orig], time_delta, time_delta

	#Not number:
	try:
		x_dt=pd.to_datetime(df[timevar])
	except ValueError as e:
		try:
			x_dt=pd.to_numeric(x_dt)
		except ValueError as e:
			raise ValueError(f"{timevar} is determined to be the date variable, but it is neither nummeric "
					"nor a date variable. Set a variable that meets these conditions as `timevar`.")
	x_dt=pd.to_numeric(x_dt)/(24*60*60*1000000000)
	x_int = x_dt.astype(int)
	if np.all(x_int==x_dt):
		x_dt = x_int

	df[timevar_orig] = df[timevar]
	df[timevar]=x_dt
	
	time_delta = get_mean_diff(df, timevar, idvar)
	time_delta_orig = get_mean_diff(df, timevar_orig, idvar)
	time_delta2 = pd.infer_freq(df[timevar])
	if time_delta_orig!=time_delta2:
		a=0

	return [timevar+ ORIG_SUFIX], time_delta, time_delta_orig

def get_mean_diff(df, timevar, idvar):
	if idvar == []:
		m = df[timevar].diff().median()
	else:
		m = df.groupby(idvar)[timevar].diff().median()

	if int(m) == m:
		m = int(m)

	if m ==0:
		raise ValueError(f'Your date variable {timevar} has zero meadian. Use another time variable or fix the one defined.')
	return m

	

def numberize_idvar(ip,df,timevar, idvar):

	if idvar==[]:
		return []
	idvar=idvar[0]
	timevar = timevar[0]

	if df.duplicated(subset=[idvar, timevar]).any():
		raise ValueError(f"Time and groups identifiers are {timevar} and {idvar}, but "
			 	   f"there are non-unique '{timevar}'-items for some or all '{idvar}'-items. "
				   f"Make sure the dates are unique and define the group and time identifiers "
				   "explicitly, if these were not those you intended. ")
	
	dtype = np.array(df[idvar]).dtype
	df[idvar + ORIG_SUFIX] = df[idvar]

	if not np.issubdtype(dtype, np.number):
		ids, ip.idvar_unique = pd.factorize(df[idvar],True)
		df[idvar]=ids
	else:
		df[idvar]=df[idvar]

	return [idvar + ORIG_SUFIX]


