#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates diagnostics and saves it to a file


import statproc as stat
import numpy as np
import regobj
import regprocs as rp
from scipy import stats as scstats
import csv
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot  as plt
import functions as fu

class diagnostics:
	def __init__(self,panel,g,G,H,robustcov_lags,ll,simple_diagnostics=False):
		"""This class calculates, stores and prints statistics and diagnostics"""
		self.panel=panel
		ll.standardize(panel)
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(panel,ll)
		
		if simple_diagnostics:
			self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(10)
			self.norm_prob=stat.JB_normality_test(panel.e_st,panel.df)			
			return
		self.reg_output,names,args,se,se_st,tstat,tsign,sign_codes=self.coeficient_output(H,G,robustcov_lags,ll)
		self.coeficient_printout(names,args,se,se_st,tstat,tsign,sign_codes)
		
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(panel,ll,10)
		self.norm_prob=stat.JB_normality_test(ll.e_st,panel)		

		self.multicollinearity_check(G)

		self.data_correlations,self.data_statistics=self.correl_and_statistics()
		
		scatterplots(panel)

		print ( 'LL: %s' %(ll.LL,))
	
		self.adf_test=stat.adf_test(panel,ll,10)
		self.save_stats(ll)
	
	def correl_and_statistics(self):
		panel=self.panel
		x_names=[]
		X=[]
		for i in panel.data.keys():
			d=panel.data[i]
			if type(d)==np.ndarray:
				x_names.append(i)
				X.append(panel.data[i])
		n=len(x_names)
		X=np.concatenate(X,1)
		x_names=np.array(x_names).reshape((1,n))
		c=stat.correl(X)
		c=np.concatenate((x_names,c),0)
		vstat=np.concatenate((np.mean(X,0).reshape((n,1)),
		                      np.std(X,0).reshape((n,1)),
		                      np.min(X,0).reshape((n,1)),
		                      np.max(X,0).reshape((n,1))),1)
		vstat=np.concatenate((x_names.T,vstat),1)
		vstat=np.concatenate(([['','Mean','SD','min','max']],vstat),0)
		x_names=np.append([['']],x_names,1).T
		c=np.concatenate((x_names,c),1)

		return c,vstat
		
	
	def coeficient_output(self,H,G,robustcov_lags,ll):
		panel=self.panel
		args=ll.args_v
		robust_cov_matrix,cov=rp.sandwich(H,G,robustcov_lags,ret_hessin=True)
		se=np.maximum(np.diag(robust_cov_matrix).flatten(),1e-200)**0.5
		se_st=np.maximum(np.diag(cov).flatten(),1e-200)**0.5
		names=np.array(panel.name_vector)

		T=len(se)
		output=[]
		tstat=np.maximum(np.minimum((args)/((se<=0)*args*1e-15+se),3000),-3000)
		tsign=1-scstats.t.cdf(np.abs(tstat),panel.df)
		sign_codes=get_sign_codes(tsign)
		
		output=np.concatenate((names.reshape((T,1)),
		                      args.reshape((T,1)),
		                      se.reshape((T,1)),
		                      se_st.reshape((T,1)),
		                      tstat.reshape((T,1)),
		                      tsign.reshape((T,1)),
		                      sign_codes.reshape((T,1))),1)
		output=np.concatenate(([['Regressors:','coef:','SE sandwich:','SE standard:','t-value:','t-sign:','sign codes:']],output),0)
		
		
		return output,names,args,se,se_st,tstat,tsign,sign_codes

	def coeficient_printout(self,names,args,se,se_st,tstat,tsign,sign_codes):
		T=len(se)
		printout=np.zeros((T,6),dtype='<U24')
		maxlen=0
		for i in names:
			maxlen=max((len(i)+1,maxlen))
		printout[:,0]=[s.ljust(maxlen) for s in names]
		
		rndlen=8
		args=np.round(args,rndlen).astype('<U'+str(rndlen))
		tstat=np.round(tstat,rndlen).astype('<U'+str(rndlen))
		se=np.round(se,rndlen).astype('<U'+str(rndlen))
		se_st=np.round(se_st,rndlen).astype('<U'+str(rndlen))
		tsign=np.round(tsign,rndlen).astype('<U'+str(rndlen))
		sep='   '
		prstr=' '*(maxlen+int(rndlen*2.7))+'SE\n'
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
		vNames=['CI:']+panel.name_vector
		k=len(vNames)-1
		matr=stat.var_decomposition(X=G,concat=True)
		matr=np.round(matr,3)
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
		    ['Number of groups:',N],
		    ['Maximum number of dates:',T],
		    ['(A, Total number of observations:',panel.NT],
		    ['(B, Observations lost to GARCH/ARIMA',panel.tot_lost_obs],		
		    ['    Total after loss of observations (A-B,:',panel.NT_afterloss],
		    ['(C, Number of Random Effects coefficients:',N],
		    ['(D, Number of Fixed Effects coefficients in the variance process:',N],
		    ['(E, Number of coefficients:',panel.len_args],
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
		add_output(output,name_list,'Correlation Matrix',self.data_correlations)
		add_output(output,name_list,'Multicollinearity',self.MultiColl)

		add_output(output,name_list,'Descriptive statistics',self.data_statistics)
		
		add_output(output,name_list,'Number of dates in each group',panel.T_arr.reshape((N,1)))
		
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
