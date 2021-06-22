#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file



import numpy as np
import stat_functions as stat
from scipy import stats as scstats
import functions as fu
import loglikelihood as logl
from tkinter import font as tkfont
import tkinter as tk
STANDARD_LENGTH=8

class statistics:
	def __init__(self,results,correl_vars=None,descriptives_vars=None,simple_statistics=False,name=None):
		"""This class calculates, stores and prints statistics and statistics"""		

		self.G=results.direction.G
		self.H=results.direction.H
		self.ll=results.ll
		self.panel=results.panel
		self.ll.standardize()
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(self.ll,True)
		self.LL_restricted=logl.LL(self.panel.args.args_restricted, self.panel).LL
		self.LL_OLS=logl.LL(self.panel.args.args_OLS, self.panel).LL		
		self.name=name
		
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(self.panel,self.ll,10)
		self.norm_prob=stat.JB_normality_test(self.ll.e_norm,self.panel)	
		self.multicollinearity_check(self.G)

		self.data_correlations,self.data_statistics=self.correl_and_statistics(correl_vars,descriptives_vars)
		
		self.adf_test=stat.adf_test(self.panel,self.ll,10)
		#self.save_stats()
			
	
	def correl_and_statistics(self,correl_vars,descriptives_vars):
		panel=self.panel
		X_names=[]
		X=[]
		correl_X,correl_names=get_variables(panel, correl_vars)
		descr_X,descr_names=get_variables(panel, descriptives_vars)
	

		c=stat.correl(correl_X)
		c=np.concatenate((correl_names,c),0)
		n=descr_X.shape[1]
		vstat=np.concatenate((np.mean(descr_X,0).reshape((n,1)),
		                      np.std(descr_X,0).reshape((n,1)),
		                      np.min(descr_X,0).reshape((n,1)),
		                      np.max(descr_X,0).reshape((n,1))),1)
		vstat=np.concatenate((descr_names.T,vstat),1)
		vstat=np.concatenate(([['','Mean','SD','min','max']],vstat),0)
		correl_names=np.append([['']],correl_names,1).T
		c=np.concatenate((correl_names,c),1)

		return c,vstat
				
	def multicollinearity_check(self,G):
		"Returns a variance decompostition matrix with headings"
		panel=self.panel
		vNames=['Max(var_proportion)','CI:']+panel.args.names_v
		k=len(vNames)-1
		matr=stat.var_decomposition(X=G,concat=True)
		matr=np.round(matr,3)
		maxp=np.max(matr[:,1:],1).reshape((matr.shape[0],1))
		matr=np.concatenate((maxp,matr),1)
		matr=np.concatenate(([vNames],matr))
		self.MultiColl=matr

	def save_stats(self):
		"""Saves the various statistics assigned to self"""
		ll=self.ll
		panel=self.panel
		N,T,k=panel.X.shape
		output=dict()
		name_list=[]
		add_output(output,name_list,'Information',[
		    ['Description:',panel.input.descr],
		    ['LL:',ll.LL],
		    ['Number of IDs:',N],
		    ['Maximum number of dates:',T],
		    ['A) Total number of observations:',panel.NT_before_loss],
		    ['B) Observations lost to GARCH/ARIMA',panel.tot_lost_obs],		
		    ['    Total after loss of observations (A-B):',panel.NT],
		    ['C) Number of Random/Fixed Effects coefficients:',N],
		    ['D) Number of Random/Fixed Effects coefficients in the variance process:',N],
		    ['E) Number of coefficients:',panel.args.n_args],
		    ['DF (A-B-C-D-E):',panel.df],
		    ['RSq:',self.Rsq],
		    ['RSq Adj:',self.Rsqadj],
		    ['LL-ratio:',self.LL_ratio],
		    ['no ac_prob:',self.no_ac_prob],
		    ['norm prob:',self.norm_prob],
		    ['ADF (dicky fuller):',self.adf_test, "1% and 5 % lower limit of confidence intervals, respectively"],
		    ['Dependent:',panel.input.Y_names]
		    ])
		
		add_output(output,name_list,'Regression',self.reg_output)
		add_output(output,name_list,'Multicollinearity',self.MultiColl)

		add_output(output,name_list,'Descriptive statistics',self.data_statistics)
		add_output(output,name_list,'Correlation Matrix',self.data_correlations)
		add_output(output,name_list,'Number of dates in each ID',panel.T_arr.reshape((N,1)))
		
		output_table=[['']]
		output_positions=['']
		for i in name_list:
			if i!='Statistics':
				output_table.extend([[''],['']])
			pos=len(output_table)+1
			output_table.extend([[i+':']])
			output_table.extend(output[i])
			output_positions.append('%s~%s~%s~%s' %(i,pos,len(output[i]),len(output[i][0])))
		output_table[0]=output_positions
		if self.name is None:
			fname=panel.input.descr.replace('\n','').replace('\r', '')
		else:
			fname=self.name
		if len(fname)>65:
			fname=fname[:30]+'...'+fname[-30:]
		fu.savevar(output_table,fname+'.csv')
		
		self.output_dict=output

	

def add_variable(name,panel,names,variables):
	if name in panel.dataframe.keys():
		d=dict(panel.dataframe[[name]])
		if type(d)==np.ndarray:
			names.append(name)
			variables.append(d)
			
def get_variables(panel,input_str):
	v=fu.split_input(input_str)
	names=[]
	variables=[]
	if not v is None:
		for i in v:
			add_variable(i, panel, names, variables)
	
	if v is None or len(names)==0:
		for i in panel.dataframe.keys():
			add_variable(i, panel, names, variables)
			
	n=len(names)
	X=np.concatenate(variables,1)
	names=np.array(names).reshape((1,n))
	return X,names
			
def add_output(output_dict,name_list,name,table):
	if type(table)==np.ndarray:
		table=np.concatenate(([[''] for i in range(len(table))],table),1)
	output_dict[name]=table
	name_list.append(name)