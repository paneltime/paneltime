#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains classes used in the regression
from .. import likelihood as logl

from . import arguments
from . import conversion
from .. import functions as fu

import numpy as np
from .. import debug
from .. import cfunctions
import time
from .. likelihood import hfunc
from .. likelihood import function as llfunc
import copy





#todo: split up panel in sub classes

class Panel:
	def __init__(self,datainput,settings):

		self.input = datainput
		self.options = copy.deepcopy(settings)
		self.dataframe = datainput.dataframe

	def init(self):
		self.initial_defs()
		self.arrayize()
		self.diff_transform()
		self.final_defs()

	def initial_defs(self):
		if np.all(np.var(self.input.Y,0)==0):
			raise RuntimeError("No variation in Y")
		p,q,d,k,m=self.options.pqdkm
		self.orig_size=len(self.input.X)
		self.max_lags=self.input.max_lags
		self.lost_obs = max((p,q,self.max_lags))+max((m,k,self.max_lags))+d#+3
		self.nW,self.nZ,self.n_beta=len(self.input.W.columns),len(self.input.Z.columns),len(self.input.X.columns)
		self.undiff=None
		if self.input.idvar is None:
			self.options.fixed_random_group_eff=0
		if self.input.idvar is None:
			self.options.fixed_random_group_eff=0		
		self.m_zero = False
		if  m==0 and k>0:
			self.m_zero = True
			k=0
			print("Warning: GARCH term removed since ARCH terms was set to 0")
		self.pqdkm=p,q,d,k,m
		self.sign_codes = [["'", 0.1], ["*", 0.05], ["**", 0.01], ["***", 0.001]]
		self.sign_codes_plain = ', '.join(f"{s[0]}p<{s[1]}" for s in self.sign_codes)
		self.sign_codes_tex = '$' + '$, $'.join(f"^{{{s[0]}}}p<{s[1]}" for s in self.sign_codes) + '$'
		self.sign_codes_tex = self.sign_codes_tex.replace("$^{'}", "'$")	




	def final_defs(self):
		
		self.W_a=self.W*self.included[3]
		self.tot_lost_obs=self.lost_obs*self.N
		self.NT=np.sum(self.included[3])
		self.NT_before_loss=self.NT
		if not hasattr(self,'n_dates'):
			self.number_of_RE_coef=0
			self.number_of_RE_coef_in_variance=0
			self.options.fixed_random_group_eff = 0
			self.options.fixed_random_time_eff = 0
			self.options.fixed_random_variance_eff = 0
		else:
			self.number_of_RE_coef=self.N*(self.options.fixed_random_group_eff>0)+self.n_dates*(self.options.fixed_random_time_eff>0)
			self.number_of_RE_coef_in_variance=(self.N*(self.options.fixed_random_group_eff>0)
					+self.n_dates*(self.options.fixed_random_time_eff>0))*(self.options.fixed_random_variance_eff>0)
		
		
		# Extracting model information by running a test run.
		llf = llfunc.LLFunction(self, -10, 10, 10)
		self.z_active = llf.z_active()
		self.h_func_cpp = hfunc.get_model_string(llf.model, -10, 1e+5,  100)
		self.h_func = llf.model.h
																
		# Adding arguments and dfs																																										+self.n_dates*(self.options.fixed_random_time_eff>0))*(self.options.fixed_random_variance_eff>0)
		self.args=arguments.arguments(self)
		self.df=self.NT-self.args.n_args-self.number_of_RE_coef-self.number_of_RE_coef_in_variance

		self.set_instrumentals()
		self.tobit()



	def diff_transform(self):
		T=self.max_T
		d=self.pqdkm[2]
		self.I=np.diag(np.ones(T))
		#differencing operator:
		if d==0:
			return
		diff_0=np.diag(np.ones(T-1),-1)
		diff_d=(self.I-diff_0)
		undiff=np.tril(np.ones((T,T)))
		for i in range(1,d):
			diff_d=fu.dot(self.I-diff_0,diff_d)
			undiff=np.cumsum(undiff,0)
		self.undiff=undiff
		#apply the :
		self.X=self.apply_diff(self.X, diff_d, d, True)
		self.Y=self.apply_diff(self.Y, diff_d, d, False)
		if not self.Z is None:
			self.Z=self.apply_diff(self.Z, diff_d, d, True)

	def apply_diff(self,X,diff_d,d,recreate_intercept):
		X_out=fu.dot(diff_d,X)*self.included[3]
		if self.input.has_intercept and recreate_intercept:
			X_out[:,:,0]=1
		X_out[:,:d]=0
		return X_out

	def params_ok(self,args):
		a=self.q_sel,self.p_sel,self.M_sel,self.K_sel
		for i in a:
			if len(i)>0:
				if np.any(np.abs(args[i])>0.999):
					return False
		return True

	def set_instrumentals(self):
		if self.input.Z.shape[1]==1: #The instrument is just a constant
			self.XIV = self.X
		else:
			self.XIV = self.Z
		return


	def subtract_means(self,X,Y,Z):
		subtract=self.options.subtract_means
		if not subtract:
			return X,Y,Z
		X=X-np.mean(X,0)
		if self.input.has_intercept:
			X[:,0]=1
		Y=Y-np.mean(Y,0)
		if self.input.Z.shape[1]==1:
			return X,Y,Z
		Z=Z-np.mean(Z,0)
		Z[:,0]=1		

	def is_single(self):
		idvar,t=self.input.idvar,self.input.timevar
		return np.all(idvar.iloc[0]==idvar) or np.all(t.iloc[0]==t)


	def arrayize(self):
		"""Convert input frames to panel arrays, observation counts and forecasts."""
		X, Y, W, idvar, timevar, timevar_orig, Z = [
			to_numpy(value) for value in (
				self.input.X, self.input.Y, self.input.W, self.input.idvar,
				self.input.timevar, self.input.timevar_orig, self.input.Z,
			)
		]
		X, Y, Z = self.subtract_means(X, Y, Z)
		self.arrayize_multi(idvar, timevar, timevar_orig, X, Y, W, Z)
		self.cross_section_count()
		self.arrayize_pred()

	def arrayize_single(self, timevar, timevar_orig, X, Y, W, Z):
		NT,k=X.shape
		self.total_obs=NT

		if not np.all(timevar[:,0]==np.sort(timevar[:,0])):#remove in production
			raise RuntimeError("The time variable is not sorted!?!")			
		self.X=np.array(X.reshape((1,NT,k)))
		self.Y=np.array(Y.reshape((1,NT,1)))
		NT,kW=W.shape
		self.W=np.array(W.reshape((1,NT,kW)))
		NT,kZ=Z.shape
		self.Z=None
		if Z.shape[1]>1:#instrumental variables used
			self.Z=np.array(Z.reshape((1,NT,kZ)))
		self.time_map=None
		self.map=np.arange(NT).reshape(1,NT)
		self.N=1
		self.max_T=NT
		self.T_arr=np.array([[NT]])
		self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
		self.date_count_mtrx=None
		self.date_count=None
		self.idincl=np.array([True])
		self.timevar = timevar_orig
		self.masking()
		self.date_map = [(0,t) for t in range(NT)]
		idvar_names = np.array(self.input.idvar_orig)[self.idvar_ix,0]
		self.remove_incomplete(idvar_names)

	def arrayize_multi(self, idvar, timevar, timevar_orig, X, Y, W, Z):
		"""Pack observed rows into zero-padded (group, observation, variable) arrays.

		The second axis is within-group row order, not a shared calendar.
		Separate date maps link observations made on the same calendar date.
		"""
		n_observations = X.shape[0]
		self.total_obs = n_observations
		self.idvar, self.idvar_ix, group_index, group_counts = np.unique(
			idvar, return_index=True, return_inverse=True, return_counts=True
		)
		idvar_names = np.array(self.input.idvar_orig)[self.idvar_ix, 0]
		n_groups = len(self.idvar)

		# Stable sorting keeps the existing chronological order within each group.
		# Integer row indices avoid an n_groups-by-n_observations Boolean matrix.
		group_order = np.argsort(group_index.reshape(-1), kind="mergesort")
		group_rows = np.split(group_order, np.cumsum(group_counts)[:-1])
		self.max_T = np.max(group_counts)
		self.idincl = group_counts >= self.lost_obs + self.options.min_group_df
		self.X = arrayize(X, n_groups, self.max_T, group_counts, self.idincl, group_rows)
		self.Y = arrayize(Y, n_groups, self.max_T, group_counts, self.idincl, group_rows)
		self.W = arrayize(W, n_groups, self.max_T, group_counts, self.idincl, group_rows)
		self.Z = arrayize(Z, n_groups, self.max_T, group_counts, self.idincl, group_rows)
		self.N = np.sum(self.idincl)

		# Map panel cells back to input rows; the inclusion mask excludes padding.
		observation_rows = np.arange(n_observations).reshape(n_observations, 1)
		self.map = arrayize(
			observation_rows, n_groups, self.max_T, group_counts,
			self.idincl, group_rows, dtype=int,
		).reshape((self.N, self.max_T))
		self.T_arr = group_counts[self.idincl].reshape((self.N, 1))
		self.date_counter = np.arange(self.max_T).reshape((self.max_T, 1))
		self.masking()
		self.map_time_to_panel_structure(
			timevar, self.N, group_counts, self.idincl, group_rows, self.included[3]
		)
		self.remove_incomplete(idvar_names)
		self.timevar = arrayize(
			timevar_orig, n_groups, self.max_T, group_counts, self.idincl, group_rows
		)
	
	def remove_incomplete(self, idvar_names):
		"""Record retained labels and report groups already excluded during packing."""
		id_name = self.input.idvar.columns[0]
		removed_ids = idvar_names[~self.idincl]
		removed_labels = formatarray(removed_ids, 90, ', ')
		if np.any(~self.idincl) and not self.options.supress_output:
			print(
				f"The following {id_name}s were removed because of insufficient observations:\n "
				f"{removed_labels}"
			)

		self.idvar_included = self.idvar[self.idincl]
		self.original_names = idvar_names[self.idincl]



	def arrayize_pred(self):
		"""Arrange regressor and variance-factor forecasts in retained-group order."""
		(
			self.X_pred, self.X_is_predicted,
			self.X_pred_timevar, self.X_pred_idvar,
		) = self.arrayize_pred_array('X')
		self.W_pred_, _, _, _ = self.arrayize_pred_array('W')

	def arrayize_pred_array(self, name):
		"""Return forecast values, availability, dates and original group labels.

		Values have shape (retained groups, horizon, variables); labels and dates
		have a singleton last axis. Groups lacking forecast rows keep zero values,
		missing labels/dates and a false availability flag.
		"""
		n_groups, _, n_variables = getattr(self, name).shape
		horizon = max(self.max_lags, 1)
		prediction_frame = getattr(self.input, name + '_pred')
		prediction_values = prediction_frame.to_numpy(dtype="float64", na_value=np.nan)
		group_codes = np.array(self.input.idvar_pred).flatten()
		group_labels = np.array(self.input.idvar_orig_pred).flatten()
		prediction_dates = np.array(self.input.timevar_orig_pred)

		values = np.zeros((n_groups, horizon, n_variables))
		dates = np.full((n_groups, horizon, 1), np.nan)
		labels = np.full((n_groups, horizon, 1), np.nan, dtype=object)
		has_prediction = np.ones(n_groups, dtype=bool)

		for row, group_code in enumerate(self.idvar[self.idincl]):
			if group_code not in group_codes:
				has_prediction[row] = False
				continue
			group_rows = group_codes == group_code
			values[row] = prediction_values[group_rows]
			dates[row] = prediction_dates[group_rows]
			labels[row] = group_labels[group_rows].reshape(horizon, 1)

		return values, has_prediction, dates, labels


	def cross_section_count(self):
		"""Count observed cells along each panel axis, retaining broadcast dimensions."""
		self.T_i = np.sum(self.included[3], axis=1).reshape((self.N, 1, 1))
		self.N_t = np.sum(self.included[3], axis=0).reshape((1, self.max_T, 1))
		# Empty slices have zero sums; a denominator of one avoids division by zero.
		self.T_i = self.T_i + (self.T_i <= 0)
		self.N_t = self.N_t + (self.N_t <= 0)


	def masking(self):
		"""Build masks and constant arrays for broadcasting over 2D through 6D data.

		Every group starts at within-group row zero. Rows beyond its observed
		length are padding, not observations. Index d selects a d-dimensional
		array, with singleton trailing axes, e.g. included[3] is (N, max_T, 1).
		"""
		included = np.array([
			self.date_counter < self.T_arr[group] for group in range(self.N)
		])
		zeros = np.zeros((self.N, self.max_T, 1))
		ones = np.ones((self.N, self.max_T, 1))
		self.included = [None, None]
		self.zeros = [None, None]
		self.ones = [None, None]

		for trailing_axes in range(5):
			shape = [self.N, self.max_T] + [1] * trailing_axes
			self.included.append(included.reshape(shape))
			self.zeros.append(zeros.reshape(shape))
			self.ones.append(ones.reshape(shape))


	def tobit(self):
		"""Sets the tobit threshold"""
		tobit_limits=self.options.tobit_limits
		if tobit_limits is None:
			return
		if len(tobit_limits)!=2:
			print("Warning: The tobit_limits argument must have lenght 2, and be on the form [floor, ceiling]. None is used to indicate no active limit")
		if (not (tobit_limits[0] is None)) and (not( tobit_limits[1] is None)):
			if tobit_limits[0]>tobit_limits[1]:
				raise RuntimeError("floor>ceiling. The tobit_limits argument must have lenght 2, and be on the form [floor, ceiling]. None is used to indicate no active limit")
		g=[1,-1]
		self.tobit_I=[None,None]
		self.tobit_active=[False, False]#lower/upper threshold
		desc=['tobit_low','tobit_high']
		for i in [0,1]:
			self.tobit_active[i]=not (tobit_limits[i] is None)
			if self.tobit_active[i]:
				if np.sum((g[i]*self.Y<g[i]*tobit_limits[i])*self.included[3]):
					print("Warning: there are observations of self.Y outside the non-censored interval. These will be ignored.")
				I=(g[i]*self.Y<=g[i]*tobit_limits[i])*self.included[3]
				self.Y[I]=tobit_limits[i]
				if np.var(self.Y)==0:
					raise RuntimeError("Your tobit limits are too restrictive. All observationss would have been cencored. Cannot run regression with these limits.")
				self.tobit_I[i]=I
				if np.sum(I)>0 and np.sum(I)<self.NT and False:#avoiding singularity #Not sure why this is here, shuld be deleted?
					self.X=np.concatenate((self.X,I),2)
					self.input.X_names.append(desc[i])

	def map_time_to_panel_structure(self, timevar, N, T_count, idincl, sel, incl):
		"""Link each observed calendar date to its (panel row, within-group row)."""
		if timevar is None:
			return None
		n_groups, n_periods, _ = incl.shape
		_, date_codes = np.unique(timevar, return_inverse=True)
		panel_date_codes = arrayize(
			np.array(date_codes).reshape((len(timevar), 1)),
			n_groups, self.max_T, T_count, idincl, sel, int,
		)

		group_rows = incl * np.arange(n_groups).reshape(n_groups, 1, 1)
		time_rows = incl * np.arange(n_periods).reshape(1, n_periods, 1)
		observed = incl[:, :, 0]
		# Each row contains (calendar-date code, panel group, within-group row).
		coordinates = np.concatenate((
			panel_date_codes[observed], group_rows[observed], time_rows[observed],
		), axis=1)
		coordinates = coordinates[np.argsort(coordinates[:, 0])]
		date_boundaries = np.flatnonzero(np.diff(coordinates[:, 0])) + 1
		rows_by_date = np.split(coordinates[:, 1:], date_boundaries)
		date_map, date_counts = self.build_time_map(rows_by_date, panel_date_codes)

		self.n_dates = len(date_counts)
		self.date_labels = np.unique(timevar)[np.unique(coordinates[:, 0]).astype(int)]
		self.date_count = np.array(date_counts).reshape(self.n_dates, 1, 1)
		self.date_map = date_map
		counts = np.asarray(date_counts, dtype=np.intp)
		self.date_starts = np.r_[0, np.cumsum(counts)[:-1]]
		self.date_flat_index = np.ravel_multi_index(
			(coordinates[:, 1].astype(np.intp), coordinates[:, 2].astype(np.intp)),
			(n_groups, n_periods),
		)
		self.date_observation_codes = np.repeat(np.arange(self.n_dates), counts)
		self.dmap_all = (
			[indices[0] for indices in date_map],
			[indices[1] for indices in date_map],
		)

	def build_time_map(self, t_map, t_arr):
		"""Return NumPy index pairs and observation counts for each nonempty date.

		`t_map` contains (group row, within-group row) pairs grouped by date.
		For example, `self.X[date_map[i]]` selects all observations on date i.
		Also fill `date_count_mtrx`, shaped like `t_arr`, with each cell's
		calendar-date count; padded cells remain zero.
		"""
		date_map = []
		date_counts = []
		self.date_count_mtrx = np.zeros(t_arr.shape)
		for date_rows in t_map:
			coordinates = np.array(date_rows).T
			if not len(coordinates):
				continue
			indices = (tuple(coordinates[0]), tuple(coordinates[1]))
			count = len(coordinates[0])
			date_map.append(indices)
			date_counts.append(count)
			self.date_count_mtrx[indices] = count

		return date_map, date_counts


	def mean(self,X,axis=None):
		dims=list(X.shape)
		dims[2:]=[1]*(len(dims)-2)
		#X=X*self.included.reshape(dims)
		if axis is None:
			return np.sum(X)/self.NT
		if axis==1:
			dims.pop(1)
			return np.sum(X,1)/self.T_i.reshape(dims)
		if axis==0:
			dims.pop(0)
			return np.sum(X,0)/self.N_t.reshape(dims)
		if axis==(0,1):
			return np.sum(np.sum(X,0),0)/self.NT

	def var(self,X,axis=None,k=1,mean=None,included=None):
		dims=list(X.shape)
		dims_m=np.array(X.shape)
		dims[2:]=[1]*(len(dims)-2)	
		if included is None:
			a=self.included[len(dims)]
		else:
			a=included
		if mean is None:
			m=self.mean(X*a, axis)
		else:
			m=mean

		if axis==None:
			Xm=(X-m)*a
			return np.sum(Xm**2)/(self.NT-k)

		if axis==1:
			dims_m[1]=1
			m=m.reshape(dims_m)
			Xm=(X-m)*a
			dims.pop(1)
			return np.sum((Xm)**2,1)/np.maximum(self.T_i-k,1).reshape(dims)
		if axis==0:
			dims_m[0]=1		
			m=m.reshape(dims_m)
			Xm=(X-m)*a
			dims.pop(0)
			return np.sum((Xm)**2,0)/np.maximum(self.N_t-k,1).reshape(dims)
		if axis==(0,1):
			dims_m[0:2]=1
			m=m.reshape(dims_m)
			Xm=(X-m)*a			
			return np.sum((Xm)**2,axis)/(self.NT-k)


def arrayize(X, N, max_T, T, id_included, selections, dtype=None):
	"""Pack 2D observations into a zero-padded (retained groups, max_T, k) array.

	`T` gives group lengths; `selections` supplies input row indices or Boolean
	masks. Excluded groups are omitted, and the inclusion mask maintained by
	Panel distinguishes padding from actual zeros.
	"""
	if X is None:
		return None

	n_variables = X.shape[1]
	if not (np.issubdtype(X.dtype, np.integer) or 
			np.issubdtype(X.dtype, np.floating)):
		try:
			X = np.array(X, dtype=float)
		except TypeError as e:
			if not 'Timestamp' in str(e):
				raise TypeError(e)

	
	if dtype is None:
		dtype = X.dtype

	packed = np.zeros((N, max_T, n_variables), dtype=dtype)
	row_index = 0
	for group, is_included in enumerate(id_included):
		if is_included:
			packed[row_index, :T[group]] = X[selections[group]]
			row_index += 1

	return packed[:row_index]





def to_numpy(x):
	if x is None:
		return None
	x=np.array(x)
	if len(x.shape)==2:
		return x
	return x.reshape((len(x),1))

def formatarray(array,linelen,sep):
	s=sep.join([str(i) for i in array])
	s='\n'.join(s[n:n + linelen] for n in range(0, len(s), linelen))	
	return s
