#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEFAULT_INTERCEPT_NAME='Intercept'
VAR_INTERCEPT_NAME='log_variance_constant'
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

	if idvar is None:
		idvar = CONST_NAME #Single group
	elif timevar is None:
		raise RuntimeError("If you have supplied an ID variable, you must also supply a time variable.")

	
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
	x=get_names(x, 'pool')
	df=df.groupy(x).agg(operation)
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

	for var in x:
		if ' ' in var:
			raise RuntimeError(f"Spaces are not allowed in variable names, but found in '{var}' from {input_type}")

		try:
			variance = np.var(dfx[var])
		except TypeError as e:
			raise TypeError(f"All variables except time and ID must be numeric. {e}")

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



def eval_and_add_pred_space(df, vars, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig):
	df = df.sort_values(idvar_orig + timevar_orig)
	df = df.set_index(idvar_orig + timevar_orig)
	df_new, lost_na_obs, max_lags, max_history, n = eval_variables(df, idvar + timevar + vars, idvar_orig)

	df_pred = get_pred_df(df, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig, max_lags, max_history, vars)

	df_new=df_new.dropna()
	

	return df_new, lost_na_obs, max_lags, n, df_pred


def get_pred_df(df, idvar_orig, timevar_orig, idvar, timevar, time_delta, time_delta_orig, max_lags, max_history, vars):
	df_pred, pred_start = extend_timeseries(df, idvar_orig, idvar, timevar_orig, timevar, time_delta, time_delta_orig, max_lags, max_history)
	df_pred, _, _, _ , _ = eval_variables(df_pred, idvar + timevar + vars, idvar_orig, False)
	#For the prediction values, collect only the last max_lags observations
	df_pred = df_pred[df_pred.index.get_level_values(timevar_orig[0]) >= pred_start]

	return df_pred




def extend_timeseries(df, idvar_orig, idvar, timevar_orig, timevar, time_delta, time_delta_orig, max_lags, max_history):
	# Ensure the data is sorted by timevar within each group


	#Works here to cut of last rows in `panel` before can be implemented.
	
	idvar_orig, idvar, timevar_orig, timevar = idvar_orig + idvar + timevar_orig + timevar

	# Find the maximum date in the dataset
	max_date = df[timevar].max()
	max_date_orig = df.index.get_level_values(timevar_orig).max()

	# Get the last date per group
	idvar_max_dates =  df.groupby(level=idvar_orig)[timevar].max()# Ensure proper column names

	# Filter groups where last date matches max_date
	extend_groups = idvar_max_dates[idvar_max_dates== max_date].index

	# Create a list to store new rows
	new_rows = []

	# Iterate over groups and time steps
	for g in extend_groups:
		
		for t in range(1, max(max_lags,1) + 1):
			new_date_orig = max_date_orig + t * time_delta_orig  # Increment original format date
			s = pd.Series(pd.NA, index=df.columns, name=(g, new_date_orig))
			s[timevar] = max_date + t * time_delta
			s[idvar] = df.loc[(g, max_date_orig), idvar]  # Copy the group identifier
			s[DEFAULT_INTERCEPT_NAME] = s[VAR_INTERCEPT_NAME] = s[INSTRUMENT_INTERCEPT_NAME] = s[CONST_NAME] = 1  # Add intercept value
			new_rows.append(s)

	new_df = pd.DataFrame(new_rows)
	new_df.index.names = df.index.names  # Ensure correct MultiIndex structure
	
	# Filter only the included groups in extended_gruops: 
	df_last = df[df.index.get_level_values(idvar_orig).isin(extend_groups)]
	# Filtering last `max_history`observations per group as only the last `max_lags` are needed
	df_last = df_last
	df_last = df_last[df_last[timevar]>max_date-(max_history+1)*time_delta]
	
	# Concatenates, workaround for the fact that one sized NA dfs are not allowed to be concatenated
	df_pred = concatenate(new_df, df_last)

	return df_pred, max_date_orig+time_delta_orig


def concatenate(new_df, df_last):
	"Concatenates, workaround for the fact that one sized NA dfs are not allowed to be concatenated"
	
	# Concatenate the new rows with the original DataFrame
	original_dtypes = df_last.dtypes.to_dict()

	# Replace dtypes with nullable equivalents where appropriate
	nullable_dtypes = {}
	for col, dtype in original_dtypes.items():
		if pd.api.types.is_integer_dtype(dtype):
			nullable_dtypes[col] = 'Int64'
		elif pd.api.types.is_float_dtype(dtype):
			nullable_dtypes[col] = 'Float64'
		else:
			nullable_dtypes[col] = 'object'

	df_last = df_last.astype('object')
	new_df = new_df.astype('object')

	new_df[new_df.isna()]='NA'
	df_last[df_last.isna()]='NA'

	df_pred = pd.concat([df_last, new_df]).sort_index()
	df_pred[df_pred=='NA'] = pd.NA

	df_pred = df_pred.astype(nullable_dtypes)

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
            - new_df (pd.DataFrame): Transformed DataFrame with evalnew_dfuated variables.
            - lost_na_obs (int): Number of observations lost due to NaN removal.
            - max_lags (int): Maximum lag required across all expressions.
            - max_history (int): Total historical data needed for computations.
            - n (int): Original number of rows in `df`.
    """

    # Initialize the transformed DataFrame
    new_df = pd.DataFrame()
    
    # Convert to panel format if `idvar_orig` is provided
    pd_panel = df.groupby(level=idvar_orig) if idvar_orig else df
    lag_obj = LagObject(pd_panel)  # Create lag object for time-series operations

    # Track original row count
    n = len(df)

    # Drop NaN values if specified
    if dropna:
        df = df.dropna()
    lost_na_obs = n - len(df)

    # Define namespaces for evaluating lagged/differenced expressions
    namespace = {'D': lag_obj.diff, 'L': lag_obj.lag, 'np': np}
    namespace_lags = {
        'D': lambda x, lag=1: diffs.append(lag),
        'L': lambda x, lag=1: lags.append(lag),
        'np': np
    }

    # Initialize tracking lists
    lags, diffs = [], []
    max_lags, max_history = [], []

    # Add all DataFrame columns to namespaces
    for column in df.columns:
        namespace[column] = df[column]
        namespace_lags[column] = 0  # Ensures column names are recognized in eval

    # Process each variable/expression in `x`
    for var in x:
        if var in df:
            # Directly copy existing columns
            new_df[var] = df[var]
        else:
            # Reset tracking lists before evaluating expressions
            lags.clear()
            diffs.clear()
            try:
                # Evaluate expression to track required lags and diffs
                eval(var, namespace_lags)
                
                # Compute and store evaluated expression
                new_df[var] = eval(var, namespace)

                # Store max lag and history depth needed
                max_lags.append(max(lags) if lags else 0)
                max_history.append(sum(lags) + sum(diffs))
            except NameError:
                raise NameError(f"{var} not defined in data frame or function")

    # Get the max required lag and total historical data needed
    max_lags = max(max_lags) if max_lags else 0
    max_history = max(max_history) if max_history else 0

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
		if inputtype=="Time variable":
			df['time']=np.arange(len(df))
			return ['time']
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
		m = df.groupby(idvar)[timevar].diff().median()
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

	if df.duplicated(subset=[idvar, timevar]).any():
		raise ValueError(f"Check your data! {timevar} and {idvar} needs to be jointly unique.\n"
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


