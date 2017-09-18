#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file


import statproc as stat
import numpy as np
import regprocs as rp
from scipy import stats as scstats
import csv
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot  as plt
import functions as fu
import loglikelihood as logl

class statistics:
	def __init__(self,results_obj,robustcov_lags=100,correl_vars=None,descriptives_vars=None,simple_statistics=False):
		"""This class calculates, stores and prints statistics and statistics"""		

		self.G=results_obj.gradient_matrix
		self.H=results_obj.hessian
		self.ll=results_obj.ll
		self.panel=results_obj.panel
		self.ll.standardize()
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(self.panel,self.ll)
		self.LL_restricted=logl.LL(self.panel.args.args_restricted, self.panel).LL
		self.LL_OLS=logl.LL(self.panel.args.args_OLS, self.panel).LL		
		(self.reg_output,
		 self.names,
		 self.args,
		 self.se_robust,
		 self.se_st,
		 self.tstat,
		 self.tsign,
		 sign_codes)=self.coeficient_output(self.H,self.G,robustcov_lags,self.ll)
		
		
		if simple_statistics:		
			return	
		self.coeficient_printout(sign_codes)
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(self.panel,self.ll,10)
		self.norm_prob=stat.JB_normality_test(self.ll.e_st,self.panel)		

		self.multicollinearity_check(self.G)

		self.data_correlations,self.data_statistics=self.correl_and_statistics(correl_vars,descriptives_vars)
		
		scatterplots(self.panel)

		print ( 'LL: %s' %(self.ll.LL,))
	
		self.adf_test=stat.adf_test(self.panel,self.ll,10)
		self.save_stats(self.ll)
	
	def correl_and_statistics(self,correl_vars,descriptives_vars):
		panel=self.panel
		x_names=[]
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
		
	
	def coeficient_output(self,H,G,robustcov_lags,ll):
		panel=self.panel
		args=ll.args_v
		robust_cov_matrix,cov=rp.sandwich(H,G,robustcov_lags,ret_hessin=True)
		se_robust=np.maximum(np.diag(robust_cov_matrix).flatten(),1e-200)**0.5
		se_st=np.maximum(np.diag(cov).flatten(),1e-200)**0.5
		names=np.array(panel.args.names_v)

		T=len(se_robust)
		output=[]
		tstat=np.maximum(np.minimum((args)/((se_robust<=0)*args*1e-15+se_robust),3000),-3000)
		tsign=1-scstats.t.cdf(np.abs(tstat),panel.df)
		sign_codes=get_sign_codes(tsign)
		
		output=np.concatenate((names.reshape((T,1)),
		                      args.reshape((T,1)),
		                      se_robust.reshape((T,1)),
		                      se_st.reshape((T,1)),
		                      tstat.reshape((T,1)),
		                      tsign.reshape((T,1)),
		                      sign_codes.reshape((T,1))),1)
		output=np.concatenate(([['Regressor:','coef:','SE sandwich:','SE standard:','t-value:','t-sign:','sign codes:']],output),0)
		
		
		return output,names,args,se_robust,se_st,tstat,tsign,sign_codes

	def coeficient_printout(self,sign_codes):
		names,args,se,se_st,tstat,tsign=self.names,self.args,self.se_robust,self.se_st,self.tstat,self.tsign
		T=len(se)
		printout=np.zeros((T,6),dtype='<U24')
		maxlen=0
		for i in names:
			maxlen=max((len(i)+1,maxlen))
		printout[:,0]=[s.ljust(maxlen) for s in names]
		
		rndlen=10
		rndlen0=8
		args=np.round(args,rndlen0).astype('<U'+str(rndlen))
		tstat=np.round(tstat,rndlen0).astype('<U'+str(rndlen))
		se=np.round(se,rndlen0).astype('<U'+str(rndlen))
		se_st=np.round(se_st,rndlen0).astype('<U'+str(rndlen))
		tsign=np.round(tsign,rndlen0).astype('<U'+str(rndlen))
		sep='   '
		prstr=' '*(maxlen+rndlen+2*len(sep)) + '_'*int(rndlen+1)+'SE'+'_'*int(rndlen)+'\n'
		prstr+='Variable names'.ljust(maxlen)[:maxlen]+sep
		prstr+='Coef'.ljust(rndlen)[:rndlen]+sep
		prstr+='sandwich'.ljust(rndlen)[:rndlen]+sep
		prstr+='standard'.ljust(rndlen)[:rndlen]+sep
		prstr+='t-stat.'.ljust(rndlen)[:rndlen]+sep
		prstr+='sign.'.ljust(rndlen)[:rndlen]+sep
		prstr+='\n'
		for i in range(T):
			b=str(args[i])
			t=str(tstat[i])
			if b[0]!='-':
				b=' '+b
				t=' '+t
			prstr+=names[i].ljust(maxlen)[:maxlen]+sep
			prstr+=b.ljust(rndlen)[:rndlen]+sep
			prstr+=se[i].ljust(rndlen)[:rndlen]+sep
			prstr+=se_st[i].ljust(rndlen)[:rndlen]+sep
			prstr+=t.ljust(rndlen)[:rndlen]+sep
			prstr+=tsign[i].ljust(rndlen)[:rndlen]+sep
			prstr+=sign_codes[i]
			prstr+='\n'
		prstr+='\n'+"Significance codes: .=0.1, *=0.05, **=0.01, ***=0.001"
		print(prstr)



				
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


	def save_stats(self,ll,strappend=''):
		"""Saves the various statistics assigned to self"""
		panel=self.panel
		N,T,k=panel.X.shape
		output=dict()
		name_list=[]
		add_output(output,name_list,'Information',[
		    ['Description:',panel.descr],
		    ['LL:',ll.LL],
		    ['Number of IDs:',N],
		    ['Maximum number of dates:',T],
		    ['A) Total number of observations:',panel.NT_before_loss],
		    ['B) Observations lost to GARCH/ARIMA',panel.tot_lost_obs],		
		    ['    Total after loss of observations (A-B):',panel.NT],
		    ['C) Number of Random Effects coefficients:',N],
		    ['D) Number of Fixed Effects coefficients in the variance process:',N],
		    ['E) Number of coefficients:',panel.len_args],
		    ['DF (A-B-C-D-E):',panel.df],
		    ['RSq:',self.Rsq],
		    ['RSq Adj:',self.Rsqadj],
		    ['LL-ratio:',self.LL_ratio],
		    ['no ac_prob:',self.no_ac_prob],
		    ['norm prob:',self.norm_prob],
		    ['ADF (dicky fuller):',self.adf_test, "1% and 5 % lower limit of confidence intervals, respectively"],
		    ['Dependent:',panel.y_name]
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
		
		fu.savevar(output_table,'output/'+panel.descr+strappend,'csv')
		
		self.output=output

def add_variable(name,panel,names,variables):
	if name in panel.dataframe.keys():
		d=panel.dataframe[name]
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
	else:
		for i in range(len(table)):
			table[i]=['']+table[i]
	output_dict[name]=table
	name_list.append(name)
	

def get_list_dim(lst):
	"""Returns 0 if not list, 1 if one dim and 2 if two or more dim. If higher than
	2 dim are attemted to print, then each cell will contain an array. Works on lists and ndarray"""
	if  type(lst)==np.ndarray:
		return min((len(lst.shape),2))
	elif type(lst)==list:
		for i in lst:
			if type(i)!=list:
				return 1
		return 2
	else:
		return 0
		
		


	
def get_sign_codes(tsign):
	sc=[]
	for i in tsign:
		if i<0.001:
			sc.append('***')
		elif i<0.01:
			sc.append('** ')
		elif i<0.05:
			sc.append('*  ')
		elif i<0.1:
			sc.append(' . ')
		else:
			sc.append('')
	sc=np.array(sc,dtype='<U3')
	return sc

def scatterplots(panel):
	
	x_names=panel.x_names
	y_name=panel.y_name
	X=panel.raw_X
	Y=panel.raw_Y
	N,k=X.shape
	for i in range(k):
		fgr=plt.figure()
		plt.scatter(X[:,i],Y[:,0], alpha=.1, s=10)
		plt.ylabel(y_name)
		plt.xlabel(x_names[i])
		xname=remove_illegal_signs(x_names[i])
		fname=fu.obtain_fname('figures/%s-%s.png' %(y_name,xname))
		fgr.savefig(fname)
		plt.close()
		
	
	
def remove_illegal_signs(name):
	illegals=['#', 	'<', 	'$', 	'+', 
	          '%', 	'>', 	'!', 	'`', 
	          '&', 	'*', 	'‘', 	'|', 
	          '{', 	'?', 	'“', 	'=', 
	          '}', 	'/', 	':', 	
	          '\\', 	'b']
	for i in illegals:
		if i in name:
			name=name.replace(i,'_')
	return name
