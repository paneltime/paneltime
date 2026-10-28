#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEFAULT_INTERCEPT_NAME='Intercept'
VAR_INTERCEPT_NAME='var_const'
INSTRUMENT_INTERCEPT_NAME='instrument_intercept'
CONST_NAME='one'
ORIG_SUFIX = '_orig'


import numpy as np
import pandas as pd
from collections import namedtuple



def get_variables(ip, df, model_string, idvar, timevar, heteroscedasticity_factors, instruments, settings, pool=(None, 'mean')):
	"""
	Processes and extracts variables for a statistical model.
	
	Args:
		ip: Internal object for storing extracted information.
		df: Pandas DataFrame containing the data.
		model_string: Model equation in the format "Y ~ X1 + X2 + X3".
		idvar: Identifier variable(s).
		timevar: Time variable(s).
		heteroscedasticity_factors: Variables used for heteroscedasticity correction.
		instruments: Instrumental variables.
		settings: Configuration object with additional options.
		pool: Tuple specifying variable and aggregation method for pooling.

	Raises:
		RuntimeError: If the DataFrame is invalid, contains duplicate column names, 
					  or the model_string is incorrectly formatted.
	"""

	model_string = model_string.replace(' ', '')
	
	if not settings.supress_output:
		print("Analyzing variables ...")

	if not isinstance(df, pd.DataFrame) or df.empty:
		raise RuntimeError("The supplied data must be a non-empty pandas DataFrame.")

	if CONST_NAME in df:
		print(f"Warning: The name '{CONST_NAME}' is reserved and will be set to 1.")

	# Ensure unique column names
	duplicated_cols = df.columns[df.columns.duplicated()]
	if duplicated_cols.any():
		raise RuntimeError(f"Duplicate column names found: '{duplicated_cols[0]}'. Column names must be unique.")

	# Resolve time and ID variables
	timevar, idvar = check_dimensions(df, timevar, idvar)
	idvar = panel_test(df, idvar, timevar)

	if idvar is None:
		idvar = CONST_NAME #Single group


	
	df = df.reset_index()

	# Apply pooling transformation
	df = pool_func(df, pool)

	# Add constant/intercept variables
	df[CONST_NAME] = 1
	df[DEFAULT_INTERCEPT_NAME] = df[VAR_INTERCEPT_NAME] = df[INSTRUMENT_INTERCEPT_NAME] = df[CONST_NAME]

	# Ensure sorting variables exist in df
	identify_sort_var(timevar, df)
	identify_sort_var(idvar, df)

	# Validate and retrieve variable names
	idvar = get_names(idvar, df, "ID variable")
	timevar = get_names(timevar, df, "Time variable")

	# Sort DataFrame by ID and Time variables (if present)
	sort_columns = idvar + timevar
	if sort_columns:
		df = df.sort_values(sort_columns)

	# Process variable groups
	W = get_names(heteroscedasticity_factors, df, "Heteroscedasticity factors", True, VAR_INTERCEPT_NAME)
	Z = get_names(instruments, df, "Instruments", True, INSTRUMENT_INTERCEPT_NAME)

	try:
		Y, X = parse_model(model_string, settings)
	except Exception:
		raise RuntimeError("The model_string must be in the format 'Y ~ X1 + X2 + X3'.")

	if not Y or Y == ['']:
		raise RuntimeError("No dependent variable specified in the model_string.")

	vars_list = W + Z + Y + X

	# Convert ID and Time variables to numeric representations
	idvar_orig = numberize_idvar(ip, df, timevar, idvar)
	timevar_orig, time_delta, time_delta_orig = numberize_time(df, timevar, idvar)

	# Prepare and extend dataset for prediction space
	df = clean_df(df, idvar + timevar + vars_list + idvar_orig + timevar_orig)
	df, ip.lost_na_obs, ip.max_lags, ip.orig_n, ip.df_pred = eval_and_add_pred_space(
		df, vars_list, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig
	)

	# Check if variables exist in the processed DataFrame
	df_test(vars_list, df)

	if df.empty:
		raise RuntimeError("Filtered dataset is empty. Check for missing values in your data.")

	# Store processed variables in `ip`
	ip.has_intercept = add_variables(ip, settings, df, ip.df_pred, locals())
	ip.dataframe = df

def panel_test(df, idvar, timevar):
	"""Tests if there is a panel structure"""
	if idvar is None or timevar is None:
		return None
	diffs = df.groupby(idvar)[timevar].diff()
	if diffs.isna().mean() > 0.9:
		return None
	else:
		return idvar

def clean_df(df, usedvars):
	vars_used = []
	for var in df:
		if (var in usedvars):
			vars_used.append(var)
		else:
			for v in usedvars:
				if var in v:
					vars_used.append(var)
					break
			
	df = df[vars_used] # Filter DataFrame to include only relevant columns

	return df

def add_variables(ip, settings, df, df_pred, locals_dict):
	"""
	Adds processed variables to the 'ip' object.

	Args:
		ip: Object to store variable data.
		settings: Configuration object with additional options.
		df: Processed dataset.
		df_pred: Extended dataset for prediction.
		locals_dict: Dictionary of local variables.

	Returns:
		bool: True if intercept was added, otherwise False.
	"""
	
	# Ensure index reset before processing
	df, df_pred = df.reset_index(), df_pred.reset_index()

	# Define variable types and properties
	VariableInfo = namedtuple("VariableInfo", ["name", "is_numeric"])
	variables = [
		VariableInfo("idvar", True),
		VariableInfo("timevar", True),
		VariableInfo("idvar_orig", False),
		VariableInfo("timevar_orig", False),
		VariableInfo("W", True),
		VariableInfo("Z", True),
		VariableInfo("Y", True),
		VariableInfo("X", True)
	]

	const_vars = {}

	for var_name, is_numeric in variables:

		var, var_pred, const_vars[var_name] = check_var(df, df_pred, locals_dict[var_name], var_name, is_numeric)

		setattr(ip, f"{var_name}_names", list(var) if var is not None else None)
		setattr(ip, var_name, var)
		setattr(ip, f"{var_name}_pred", var_pred)

	return const_vars["X"]

		
		

def check_dimensions(df, timevar, idvar):
	"""
	Determines the time variable (`timevar`) and ID variable (`idvar`) in a MultiIndex DataFrame.

	Parameters:
		df (pd.DataFrame): The input DataFrame with a MultiIndex.
		timevar (str or None): The name of the time variable (if known).
		idvar (str or None): The name of the ID variable (if known).

	Returns:
		tuple: (timevar, idvar) - Identified names of the time variable and ID variable.
	"""
	ix = df.index  # Get the DataFrame's index

	# If `timevar` is already provided or index is a simple RangeIndex, return as-is
	if (timevar is not None) or isinstance(ix, pd.RangeIndex):
		return timevar, idvar

	# Loop through all levels of the MultiIndex
	for k in range(len(ix.names)):
		try:
			# Try converting the k-th index level to datetime format
			pd.to_datetime(ix.get_level_values(k), format='%Y-%m-%d')

			# If successful, assume this is the time variable
			names = list(ix.names)  # Get MultiIndex level names
			timevar = names.pop(k)  # Remove the detected time variable

			# If `idvar` is not provided, try to infer it
			if idvar is None:
				unique_dates = len(ix.get_level_values(timevar))  # Count unique time points
				unique_indices = len(set(df.index.to_list()))  # Count total unique index tuples

				# If unique dates ≠ unique index tuples, assume first remaining index is an ID variable
				if unique_dates != unique_indices:
					idvar = names[0] if names else None  # Assign first remaining level as ID var

			return timevar, idvar  # Return once a time variable is found
		except Exception:
			# If conversion fails, continue checking other index levels
			pass

	# If no time variable is found, return inputs unchanged
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
	x=get_names(x, df, 'pool')
	df=df.groupby(x).agg(operation)
	return df



def check_var(df, df_pred, x, input_type, numeric):
	if not x: 
		return None, None, None

	dfx = df[x]
	if df_pred.empty:
		dfx_pred = None
	else:
		dfx_pred = df_pred[x]


	if not numeric:
		return dfx, dfx_pred, None

	const_found = False

	for i, var in enumerate(x):
		x[i].replace(' ','')
		try:
			variance = np.var(dfx[var])
		except TypeError as e:
			raise TypeError(f"All variables except time and ID must be numeric. {e}")
		except RuntimeWarning as e:
			raise TypeError(f"{var} contains invalid values. {e}")

		if variance == 0:
			if const_found:
				msg = f"Warning: {var} from {input_type} is constant. Variable dropped."
				if dfx[var].iloc[0] == 0:
					msg = f"Warning: All values in '{var}' from {input_type} are zero. Variable dropped."
				print(msg)

				dfx = dfx.drop(columns=[var])
				if dfx_pred is not None:
					dfx_pred = dfx_pred.drop(columns=[var])
			else:
				if input_type == 'Y':
					raise RuntimeError("The dependent variable is constant")
				const_found = True

	return dfx, dfx_pred, const_found



def eval_and_add_pred_space(
	df, vars, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig,
):
	"""Evaluate estimation data and prepare a separate future prediction frame."""
	df = df.sort_values(idvar_orig + timevar_orig)
	df = df.set_index(idvar_orig + timevar_orig)
	variable_names = idvar + timevar + vars
	evaluated, lost_na_obs, max_lags, max_history, n = eval_variables(
		df, variable_names, idvar_orig
	)
	df_pred = get_pred_df(
		df, idvar_orig, timevar_orig, idvar, timevar,
		time_delta, time_delta_orig, max_lags, max_history, vars,
	)

	# Lagged expressions can introduce missing values after input filtering.
	return evaluated.dropna(), lost_na_obs, max_lags, n, df_pred


def get_pred_df(
	df, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig,
	max_lags, max_history, vars,
):
	"""Evaluate expressions with their history, then retain only future rows."""
	extended, pred_start = extend_timeseries(
		df, idvar_orig, idvar, timevar_orig, timevar,
		time_delta, time_delta_orig, max_lags, max_history,
	)
	evaluated, _, _, _, _ = eval_variables(
		extended, idvar + timevar + vars, idvar_orig, dropna=False
	)
	dates = evaluated.index.get_level_values(timevar_orig[0])
	return evaluated[dates >= pred_start]


def extend_timeseries(
	df, idvar_orig, idvar, timevar_orig, timevar, time_delta, time_delta_orig,
	max_lags, max_history,
):
	"""Append future rows for groups observed at the final date.

	Keep enough preceding rows to evaluate lagged/differenced expressions.
	Original ID/date labels live in the index; numeric codes live in columns.
	Even formulas without lags receive one prediction row per eligible group.
	"""
	group_level, group_column = idvar_orig[0], idvar[0]
	date_level, date_column = timevar_orig[0], timevar[0]
	last_date = df[date_column].max()
	last_date_orig = df.index.get_level_values(date_level).max()
	last_date_by_group = df.groupby(level=group_level)[date_column].max()
	extend_groups = last_date_by_group[last_date_by_group == last_date].index

	# The product index is group-major: repeat IDs and tile future dates.
	steps = np.arange(1, max(max_lags, 1) + 1)
	new_dates = [last_date_orig + int(step) * time_delta_orig for step in steps]
	new_index = pd.MultiIndex.from_product(
		[extend_groups, new_dates], names=df.index.names
	)
	future = pd.DataFrame(pd.NA, index=new_index, columns=df.columns)
	future[date_column] = np.tile(last_date + steps * time_delta, len(extend_groups))
	last_rows = [(group, last_date_orig) for group in extend_groups]
	group_codes = df.loc[last_rows, group_column].to_numpy()
	future[group_column] = np.repeat(group_codes, len(steps))
	for name in (
		DEFAULT_INTERCEPT_NAME, VAR_INTERCEPT_NAME, INSTRUMENT_INTERCEPT_NAME, CONST_NAME,
	):
		future[name] = 1

	history = df[df.index.get_level_values(group_level).isin(extend_groups)]
	history_start = last_date - (max_history + 1) * time_delta
	history = history[history[date_column] > history_start]

	return concatenate(future, history), last_date_orig + time_delta_orig


def concatenate(new_df, df_last):
	"""Combine history and future rows without modifying either input.

	Nullable numeric types allow unknown future values without converting
	integer columns to floats. Other columns retain object-valued labels.
	"""
	nullable_dtypes = {}
	for col, dtype in df_last.dtypes.to_dict().items():
		if pd.api.types.is_integer_dtype(dtype):
			nullable_dtypes[col] = 'Int64'
		elif pd.api.types.is_float_dtype(dtype):
			nullable_dtypes[col] = 'Float64'
		else:
			nullable_dtypes[col] = 'object'

	df_pred = pd.concat([
		df_last.astype(nullable_dtypes),
		new_df.astype(nullable_dtypes),
	]).sort_index()
	# Normalize object-column missing values to the same sentinel as numerics.
	for col, dtype in nullable_dtypes.items():
		if dtype == 'object':
			df_pred[col] = df_pred[col].where(df_pred[col].notna(), pd.NA)

	return df_pred

def convert_datetime_columns_to_float(df):
	df_converted = df.copy()
	for col in df_converted.columns:
		if np.issubdtype(df_converted[col].dtype, np.datetime64):
			df_converted[col] = df_converted[col].astype('int64') / 1e9  # Convert nanoseconds to seconds
		elif pd.api.types.is_object_dtype(df_converted[col]):
			try:
				df_converted[col] = pd.to_datetime(df_converted[col])
				df_converted[col] = df_converted[col].astype('int64') / 1e9
			except (ValueError, TypeError):
				pass  # Leave non-datetime strings unchanged
	return df_converted.astype('Float64')

def eval_variables(df, x, idvar_orig, dropna=True):
	"""
	Evaluates variables by handling lags and differences in a DataFrame.

	Parameters:
		df (pd.DataFrame): Input DataFrame (potentially with MultiIndex).
		x (list): List of variable names (including lagged/differenced expressions).
		idvar_orig (list): ID variable(s) for panel data grouping.
		dropna (bool): Whether to drop NaN values from the dataset.

	Returns:
		tuple: (new_df, lost_na_obs, max_lags, max_history, n)
			- new_df (pd.DataFrame): Evaluated variables, possibly with lag-induced NaNs.
			- lost_na_obs (int): Rows removed before expression evaluation.
			- max_lags (int): Largest individual L() lag, used as forecast horizon.
			- max_history (int): Largest sum of L()/D() lags in one expression.
			- n (int): Original number of rows in `df`.
	"""

	n = len(df)
	original_df = df
	if dropna:
		df = df.dropna()
	lost_na_obs = n - len(df)

	# Preserve first-occurrence order while matching repeated column assignment.
	if x and all(var in df.columns for var in x):
		new_df = df[list(dict.fromkeys(x))].copy()
		new_df.columns.name = None
		return new_df, lost_na_obs, 0, 0, n

	new_df = pd.DataFrame()
	pd_panel = original_df.groupby(level=idvar_orig) if idvar_orig else original_df
	lag_obj = LagObject(pd_panel)

	lags, diffs = [], []
	lag_depths, history_depths = [], []
	namespace = {'D': lag_obj.diff, 'L': lag_obj.lag, 'np': np}
	# A scalar-only pass records requested lags without evaluating full Series.
	namespace_lags = {
		'D': lambda x, lag=1: diffs.append(lag),
		'L': lambda x, lag=1: lags.append(lag),
		'np': np
	}

	for column in df.columns:
		namespace[column] = df[column]
		namespace_lags[column] = df[column].iloc[0]

	for var in x:
		if var in df:
			new_df[var] = df[var]
		else:
			lags.clear()
			diffs.clear()
			try:
				eval(var, namespace_lags)
				new_df[var] = eval(var, namespace)
				lag_depths.append(max(lags) if lags else 0)
				history_depths.append(sum(lags) + sum(diffs))
			except NameError:
				raise NameError(f"{var} not defined in data frame or function")
			except Exception as e:
				raise RuntimeError(f"Problem with variable '{var}': {e}")

	max_lags = max(lag_depths) if lag_depths else 0
	max_history = max(history_depths) if history_depths else 0
	return new_df, lost_na_obs, max_lags, max_history, n



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
	X = split_regression(X)
	Y = Y.strip()
	if X==['']:
		X=[DEFAULT_INTERCEPT_NAME]
	if settings.add_intercept and not (DEFAULT_INTERCEPT_NAME in X):
		X=[DEFAULT_INTERCEPT_NAME]+X
	return [Y], ordered_unique(X)

def split_regression(expr: str):
    parts = []
    buf = []
    depth = 0  # parentheses nesting level
    
    for ch in expr:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == '+' and depth == 0:
            # split point only if not inside parentheses
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    
    if buf:  # last part
        parts.append(''.join(buf).strip())
    
    return parts

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
		if inputtype=="Time variable":
			df['time']=np.arange(len(df))
			return ['time']
	elif isinstance(x, str):
		r=[x]
	elif isinstance(x, (list, tuple)):
		r=list(x)
	
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

	# if number:
	dtype = np.array(df[timevar]).dtype

	if np.issubdtype(dtype, np.number):
		df[timevar + ORIG_SUFIX] = df[timevar]
		time_delta = get_mean_diff(df, timevar, idvar)
		if time_delta is None:
			return [],None, None
		if np.issubdtype(dtype, np.integer):
			time_delta = int(time_delta)
			if time_delta == 0:
				time_delta = 1
		return [timevar_orig], time_delta, time_delta

	# not number:
	try:
		x_dt=pd.to_datetime(df[timevar])
	except ValueError as e:
		try:
			x_dt=pd.to_numeric(x_dt)
		except ValueError as e:
			raise ValueError(f"{timevar} is determined to be the date variable, but it is neither nummeric "
					"nor a date variable. Set a variable that meets these conditions as `timevar`.")
		x_dt=pd.to_numeric(x_dt)/(24*60*60*1000000000)

	df[timevar_orig] = df[timevar]
	if x_dt.dtype == '<M8[ns]':
		df[timevar_orig] = x_dt
	df[timevar]=x_dt.astype('int64')/ 1e9
	
	time_delta = get_mean_diff(df, timevar, idvar)
	time_delta_orig = get_mean_diff(df, timevar_orig, idvar)

	return [timevar_orig], time_delta, time_delta_orig


def datetime_to_float_year(d):
	year_start = pd.to_datetime(f'{d.year}-01-01')
	next_year_start = pd.to_datetime(f'{d.year + 1}-01-01')
	return d.year + ((d - year_start).total_seconds() / (next_year_start - year_start).total_seconds())



def get_mean_diff(df, timevar, idvar):
	if idvar == []:
		m = df[timevar].diff().median()
	else:
		diffs = df.groupby(idvar)[timevar].diff()
		if diffs.isna().mean() > 0.9:
			return
		m = diffs.median()
	try:
		if int(m) == m:
			m = int(m)
	except TypeError:
		pass

	if m ==0:
		raise ValueError(f'Your date variable {timevar} has zero meadian. Use another time variable or fix the one defined.')
	return m

	

def numberize_idvar(ip,df,timevar, idvar):
	idvar=idvar[0]
	timevar = timevar[0]
	duplicated = df.duplicated(subset=[idvar, timevar])
	if duplicated.any():
		print(f"Warning: Check your data! {timevar} and {idvar} needs to be jointly unique.\n"
			 	   f"there are non-unique '{timevar}'-items for some or all '{idvar}'-items. "
				   f"Following dupliates are deleted ({sum(duplicated)} items). "
				   f"\n{df[duplicated][[idvar, timevar]]}"
				   )	
		df.drop_duplicates(subset=[idvar, timevar], inplace=True)
	
	dtype = np.array(df[idvar]).dtype
	df[idvar + ORIG_SUFIX] = df[idvar]

	if not np.issubdtype(dtype, np.number):
		ids, ip.idvar_unique = pd.factorize(df[idvar],True)
		df[idvar]=ids
	else:
		df[idvar]=df[idvar]

	return [idvar + ORIG_SUFIX]
