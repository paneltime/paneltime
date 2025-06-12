#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object

#for debug. comment out!
import matplotlib.pyplot as plt

from ..output import stat_functions
from .. import random_effects as re
from .. import functions as fu
from . import function
from ..output import stat_dist
from ..processing import model_parser
from . import arma

import numpy as np
import traceback
import sys
import time
import pandas as pd



class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
	that store dynamic variables that depend on the \n
	If args is a dictionary, the ARMA-GARCH orders are 
	determined from the dictionary. If args is a vector, the ARMA-GARCH order needs to be consistent
	with the  panel object
	"""
	def __init__(self,args,panel,constraints=None,print_err=False):
		self.err_msg = ''
		self.errmsg_h = ''

		#checking settings. If the FE/RE is done on the data before LL
		gfre=panel.options.fixed_random_group_eff
		tfre=panel.options.fixed_random_time_eff
		vfre=panel.options.fixed_random_variance_eff

		self.re_obj_i=re.re_obj(panel,True,panel.T_i,panel.T_i,gfre)
		self.re_obj_t=re.re_obj(panel,False,panel.date_count_mtrx,panel.date_count,tfre)
		self.re_obj_i_v=re.re_obj(panel,True,panel.T_i,panel.T_i,gfre*vfre)
		self.re_obj_t_v=re.re_obj(panel,False,panel.date_count_mtrx,panel.date_count,tfre*vfre)

		self.args=panel.args.create_args(args,panel,constraints)
		self.h_err=""
		self.LL=None
		self.LL=self.LL_calc(panel) #For debugging
		try:
			self.LL=self.LL_calc(panel)
			if np.isnan(self.LL):
				self.LL=None						
		except Exception as e:
			if print_err:
				traceback.print_exc()
				print(self.errmsg_h)




	def LL_calc(self,panel):
		X=panel.XIV
		N, T, k = X.shape
		incl = panel.included[3]

		
		G = fu.dot(panel.W_a, self.args.args_d['omega'])
		G[:,0,0] = panel.args.init_var
		if True:
			if 'initvar' in self.args.args_d:
				G[:,0,0] = self.args.args_d['initvar'][0][0]

		#Idea for IV: calculate Z*u throughout. Mazimize total sum of LL. 
		u = panel.Y-fu.dot(X,self.args.args_d['beta'])
		u_RE = (u+self.re_obj_i.RE(u, panel)+self.re_obj_t.RE(u, panel))*incl


		matrices=self.arma_calc(panel, u_RE, G)

		if matrices is None:
			return None		
		AMA_1,AMA_1AR,GAR_1,GAR_1MA, e_RE, var, h=matrices


		z = getattr(self.args.args_d, 'z', None)
		self.llfunc = function.LLFunction(panel,  e_RE, var, z)
		
		if False:#debug
			from .. import debug
			if np.any(h!=self.llfunc.h_val):
				print('the h calculated in the c function and the self.h_val calcualted here do not match')
			debug.test_c_armas(u_RE, var, e_RE, panel, self, G)

		self.variance_RE(panel,self.llfunc.e2)

		for k in function.HFUNC_ITEMS:
			setattr(self, k, getattr(self.llfunc, k))

		ll_value = self.llfunc.ll()


		self.tobit(panel,ll_value)
		LL=np.sum(ll_value*incl)

		self.LL_all=np.sum(ll_value)

		#self.add_variables_old(panel,matrices, u, u_RE, var, v, G,e_RE,e2,v_inv,LL_full)
		self.add_variables(panel, u, u_RE, G,e_RE,self.llfunc.e2, var)
		

		if abs(LL)>1e+100: 
			return None				
		return LL
	



			
	def add_variables(self,panel,u, u_RE,G,e_RE,e2, var):

		self.e_norm=e_RE*self.llfunc.v_inv05	
		self.e_RE_norm_centered=(self.e_norm-panel.mean(self.e_norm))*panel.included[3]
		self.u, self.u_RE      = u,  u_RE
		self.G=G
		self.e_RE=e_RE
		self.e2=e2
		self.ll_value = self.llfunc.ll_value
		self.var = var

	def tobit(self,panel,LL):
		if sum(panel.tobit_active)==0:
			return
		g=[1,-1]
		self.F=[None,None]	
		for i in [0,1]:
			if panel.tobit_active[i]:
				I=panel.tobit_I[i]
				self.F[i]= stat_dist.norm(g[i]*self.e_norm[I])
				LL[I]=np.log(self.F[i])


	def variance_RE(self,panel,e2):
		"""Calculates random/fixed effects for variance."""
		#not in use, expermental. Intended for a variance version of RE/FE
		self.vRE,self.varRE,self.dvarRE=panel.zeros[3],panel.zeros[3],panel.zeros[3]
		self.ddvarRE,self.dvarRE_mu,self.ddvarRE_mu_vRE=panel.zeros[3],None,None
		self.varRE_input, self.ddvarRE_input, self.dvarRE_input = None, None, None
		return

	def get_re(self, panel, x = None):
		if x == None:
			x = self.u
		return self.re_obj_i.RE(x, panel), self.re_obj_t.RE(x, panel)


	def standardize(self,panel,reverse_difference=False):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self. 
		If reverse_difference and the ARIMA difference term d>0, the standardized variables are converted to
		the original undifferenced order. This may be usefull if the predicted values should be used in another 
		differenced regression."""

		# currently seems to be some confusion with what is standardization and 
		# inverting the ARIMA transformation. Need to use inverted operators to invert
		# back to original residuals.

		# Will then simply invert self.AMA_1AR


		if hasattr(self,'Y_st') and False:
			return		
		m=panel.lost_obs
		N,T,k=panel.X.shape
		if model_parser.DEFAULT_INTERCEPT_NAME in panel.args.caption_d['beta']:
			m=self.args.args_d['beta'][0,0]
		else:
			m=panel.mean(panel.Y)	
		#e_norm=self.standardize_variable(panel,self.u,reverse_difference)
		self.Y_long = panel.input.Y
		self.X_long = panel.input.X
		self.Y_st,   self.Y_st_long   = self.standardize_variable(panel,panel.Y,reverse_difference)
		self.X_st,   self.X_st_long   = self.standardize_variable(panel,panel.X,reverse_difference)
		self.XIV_st, self.XIV_st_long = self.standardize_variable(panel,panel.XIV,reverse_difference)
		self.Y_fitted_st=fu.dot(self.X_st,self.args.args_d['beta'])
		self.Y_fitted=fu.dot(panel.X,self.args.args_d['beta'])	
		self.e_RE_norm_centered_long=self.stretch_variable(panel,self.e_RE_norm_centered)
		self.Y_fitted_st_long=self.stretch_variable(panel,self.Y_fitted_st)
		self.Y_fitted_long=np.dot(panel.input.X,self.args.args_d['beta'])
		self.u_long=np.array(panel.input.Y-self.Y_fitted_long)
		
		a=0


	def standardize_variable(self,panel,X,norm=False,reverse_difference=False):
		X=fu.arma_dot(self.AMA_1AR,X,self)
		X=(X+self.re_obj_i.RE(X, panel,False)+self.re_obj_t.RE(X, panel,False))
		if (not panel.undiff is None) and reverse_difference:
			X=fu.dot(panel.undiff,X)*panel.included[3]		
		if norm:
			X=X*self.llfunc.v_inv05
		X_long=self.stretch_variable(panel,X)
		return X,X_long		

	def stretch_variable(self,panel,X):
		N,T,k=X.shape
		m=panel.map
		NT=panel.total_obs
		X_long=np.zeros((NT,k))
		X_long[m]=X
		return X_long



	def copy_args_d(self):
		return copy_array_dict(self.args.args_d)


	def arma_calc(self,panel, u, G):
		matrices = arma.set_garch_arch(panel,self.args.args_d, u, G)
		self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA, self.e, self.var, self.h = matrices
		self.AMA_dict={'AMA_1':None,'AMA_1AR':None,'GAR_1':None,'GAR_1MA':None}		
		return matrices
	
	def predict(self, panel):
		self.standardize(panel)
		d = self.args.args_d

		N, T, k = panel.X.shape
		self.u_pred = pred_u(self.u, self.e, d['rho'], d['lambda'], panel)
		#u_pred = pred_u(self.u[:,:-1], self.e[:,:-1], d['rho'], d['lambda'], panel)#test
		self.y_pred = pred_y(panel.X, panel.X_pred[:,0], d['beta'], self.u_pred, d['rho'], d['lambda'], panel)
		self.var_pred = pred_var(self.h, self.var, d['psi'], d['gamma'], d['omega'], self.minvar, self.maxvar, panel)
		#var_pred = pred_var(self.h[:,:-1], self.var[:,:-1], d['psi'], d['gamma'], d['omega'], W, self.minvar, self.maxvar, panel)#test
		if not hasattr(self,'Y_fitted'):
			self.standardize()
		index = pd.MultiIndex.from_arrays(
				[panel.X_pred_idvar[:,0,0].flatten(), panel.X_pred_timevar[:,0,0].flatten()],  # Flatten if necessary
				names=[panel.input.idvar_names[0], panel.input.timevar_names[0]]
			)
		pred_df = pd.DataFrame({
			f'Predicted {panel.input.Y_names[0]}': self.y_pred.flatten(),
			'Predicted variance':self.var_pred.flatten(), 
			'Predicted residual':self.u_pred.flatten(), 	
							}, index=index)
		pred_df = pred_df[~np.isnan(index.get_level_values(1))]
		incl = panel.included[2]
		index = pd.MultiIndex.from_arrays(
				[np.array(T*[panel.original_names]).T[incl] , panel.timevar[incl][:,0]],  # Flatten if necessary
				names=[panel.input.idvar_names[0], panel.input.timevar_names[0]]
			)
		
		fit_df = pd.DataFrame({
			f'Observed {panel.input.Y_names[0]}': panel.Y[incl][:,0], 
			f'Fitted {panel.input.Y_names[0]}': self.Y_fitted[incl][:,0], 
			'Fitted residual':self.u[incl][:,0],
			'Fitted variance': self.llfunc.v[incl][:,0] 	
							}, index=index)
		

		df = pd.concat((pred_df, fit_df), axis=1)
		return df.sort_index()

def get_last_obs(u, panel):
	maxlag = max(panel.pqdkm)
	N,T,k = panel.X.shape
	u_new = np.zeros((N,maxlag,1))
	u_new2 = np.zeros((N,maxlag,1))
	for t in range(maxlag):
		u_new[:, maxlag-1-t] = u[(np.arange(N), panel.T_arr[:,0]-1-t)]
	u_new[panel.X_is_predicted==False] = np.nan
	return u_new

def pred_y(X, x_pred_lags, beta, u_pred, rho, lmbda, panel, e_now = 0):
	N,T,k = X.shape
	x_pred_extrpolate = pred_x(X, panel)
	#Substitutes first row
	x_pred_lags[np.isnan(x_pred_lags)] = x_pred_extrpolate[np.isnan(x_pred_lags)]

	y_pred = np.sum(x_pred_lags*beta.T, axis=1).reshape((N,1))
	y_pred_ag = y_pred + u_pred
	
	return y_pred_ag


def pred_x(X, panel):
	"""
	Predicts values for a given NxTxk matrix (X) using a time-dependent 
	regression model that includes lagged variables and time trends.

	Parameters:
	X (numpy.ndarray): An NxTxk matrix where:
		- N is the number of groups.
		- T is the number of time periods.
		- k is the number of variables (including the dependent variable).
	panel: An object containing necessary panel attributes, specifically:
		- panel.included[3]: A mask for selecting relevant elements.
		- panel.T_arr: An Nx1 array indicating the time indices.
		- panel.X_is_predicted: A boolean mask specifying which values should be predicted.

	Returns:
	numpy.ndarray: An Nxkx1 matrix containing the predicted values for each group and variable.
				   Predictions are constrained within the observed range of X.
				   Unpredicted values are set to NaN.
	"""
	N, T, k = X.shape

	tarr = (np.ones((N,1))*np.arange(T)).reshape((N,T,1))*panel.included[3]
	pred = np.zeros((N, k))

	for i in range(1, k):
		x = X[:,:,i:i+1]
		xlag = np.roll(x, 1)*panel.included[3]
		z = np.concatenate((np.ones((N,T,1)), 
					  		tarr, 
							xlag), axis=2)*panel.included[3]
		
		new_z  = np.concatenate((np.ones((N, 1)), 
						   		(panel.T_arr), 
								x[(np.arange(N), panel.T_arr[:,0]-1)]), axis=1)
		coefs = np.zeros((N,z.shape[2]))
		
		for j in range(N):
			coefs[j] = (np.linalg.pinv(z[j].T @ z[j]) @ z[j].T @ x[j])[:,0]
		pred[:,i] = np.sum(new_z * coefs, axis=1)
		pred[:,i] = np.clip(pred[:,i], np.min(x, axis=(1,2)), np.max(x, axis=(1,2)))
	pred[:,0] = 1
	pred[panel.X_is_predicted==False] = np.nan
	return pred

def pred_u(u, e, rho, lmbda, panel, e_now = 0):
	pred = pred_mean(u, e, rho, lmbda, panel, e_now)	
	pred[panel.X_is_predicted==False] = np.nan
	return pred

def pred_mean(u, e, rho, lmbda, panel, e_now = 0):
	if len(lmbda)==0 and len(rho)==0:
		return u[:,-1]*0
	u_pred = e_now
	u_last = get_last_obs(u, panel)
	e_last = get_last_obs(e, panel)
	if len(rho):
		u_pred += sum([
			rho[i]*u_last[:,-i-1] for i in range(len(rho))
			])
	if len(lmbda):
		u_pred += sum([
			lmbda[i]*e_last[:,-i-1] for i in range(len(lmbda))
		])  
	#if len(u_pred)==1:
	#	u_pred = u_pred[0,0]
	return u_pred
	
def pred_var(h, var, psi, gamma, omega, minvar, maxvar, panel):
	W = pred_x(panel.W, panel)
	G = np.dot(W,omega)
	a, b = 0, 0 
	h_last = get_last_obs(h, panel)
	var_last = get_last_obs(var, panel)
	if len(psi):
		a = sum([
			psi[i]*h_last[:,-i-1] for i in range(len(psi))
			])
	if len(gamma):
		b = sum([
			gamma[i]*(var_last[:,-i-1]) for i in range(len(gamma))
		])  
		
	var_pred = G + a +b
	var_pred = np.maximum(np.minimum(var_pred, maxvar), minvar)

	var_pred[panel.X_is_predicted==False] = np.nan
	return var_pred



def test_variance_signal(W, h, omega):
	if W is None:
		return None
	N,T,k= h.shape
	if N==1:
		W = W.flatten()
		if len(W)!=len(omega):
				raise RuntimeError("The variance signals needs to be a numpy array of numbers with "
													 "the size equal to the HF-argument variables +1, and first variable must be 1")
		return W.reshape((1,len(omega)))
	else:
		try:
			NW,kW = W.shape
			if NW!=N or kW!=k:
				raise RuntimeError("Rows and columns in variance signals must correspond with"
													 "the number of groups and the size equal to the number of "
													 "HF-argument variables +1, respectively")
		except:
			raise RuntimeError("If there are more than one group, the variance signal must be a matrix with"
												 "Rows and columns in variance signals must correspond with"
												 "the number of groups and the size equal to the number of "
													 "HF-argument variables +1, respectively"                       )      
	return W
	







def copy_array_dict(d):
	r=dict()
	for i in d:
		r[i]=np.array(d[i])
	return r
