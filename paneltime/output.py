#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file


import stat_functions as stat
import numpy as np
from scipy import stats as scstats
import csv
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot  as plt
import functions as fu
import calculus_functions as cf
import loglikelihood as logl

class statistics:
	def __init__(self,results,robustcov_lags=100,correl_vars=None,descriptives_vars=None,simple_statistics=False):
		"""This class calculates, stores and prints statistics and statistics"""		

		self.G=results.gradient_matrix
		self.H=results.hessian
		self.ll=results.ll
		self.panel=results.panel
		self.ll.standardize()
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(self.panel,self.ll)
		self.LL_restricted=logl.LL(self.panel.args.args_restricted, self.panel).LL
		self.LL_OLS=logl.LL(self.panel.args.args_OLS, self.panel).LL		
		
		
		
		if simple_statistics:	
			self.output=self.arrange_output(robustcov_lags,results.constraints,direction=results.dx_norm)
			return	
		
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(self.panel,self.ll,10)
		self.norm_prob=stat.JB_normality_test(self.ll.e_st,self.panel)		
		self.output=self.arrange_output(robustcov_lags,results.constraints,self.norm_prob,self.no_ac_prob,direction=results.dx_norm)
		self.reg_output=self.output.outputmatrix
		self.multicollinearity_check(self.G)

		self.data_correlations,self.data_statistics=self.correl_and_statistics(correl_vars,descriptives_vars)
		
		scatterplots(self.panel)

		self.adf_test=stat.adf_test(self.panel,self.ll,10)
		self.save_stats()
		
	def arrange_output(self,robustcov_lags,constraints,norm_prob,ac_prob,direction):
		panel,H,G,ll=self.panel,self.H,self.G,self.ll
		l=10
		pr=[['names','namelen',False,'Variable names',False,False],
	        ['args',l,True,'Coef',True,False],
	        ['se_robust',l,True,'sandwich',False,False],
	        ['se_st',l,True,'standard',False,False],
	        ['tstat',l,True,'t-stat.',True,False],
	        ['tsign',l,True,'sign.',False,False],
	        ['sign_codes',4,False,'',False,True],
		    ['direction',l,True,'last direction',True,False],
		    ['set_to',6,False,'set to',False,True],
		    ['assco',20,False,'associated variable',False,True],
		    ['cause',l,False,'cause',False,False]]
		o=output(pr, panel, H, G, robustcov_lags, ll, constraints,direction=direction)
		o.add_heading(top_header=' '*40 + '_'*11+'SE'+'_'*11+" "*48+"_"*9+"restricted variables"+"_"*9,
		              statistics= [['Normality',norm_prob,3,'%'], 
		                           ['P(no AC)',ac_prob,3,'%']])
		o.add_footer("Significance codes: .=0.1, *=0.05, **=0.01, ***=0.001")
		return o	
	
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
		    ['Description:',panel.descr],
		    ['LL:',ll.LL],
		    ['Number of IDs:',N],
		    ['Maximum number of dates:',T],
		    ['A) Total number of observations:',panel.NT_before_loss],
		    ['B) Observations lost to GARCH/ARIMA',panel.tot_lost_obs],		
		    ['    Total after loss of observations (A-B):',panel.NT],
		    ['C) Number of Random Effects coefficients:',N],
		    ['D) Number of Fixed Effects coefficients in the variance process:',N],
		    ['E) Number of coefficients:',panel.args.n_args],
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
		
		fu.savevar(output_table,panel.descr+'.csv')
		
		self.output_dict=output
		
def t_stats(panel,args,H,G,robustcov_lags,):

	robust_cov_matrix,cov=sandwich(H,G,robustcov_lags,ret_hessin=True)
	se_robust=np.maximum(np.diag(robust_cov_matrix).flatten(),1e-200)**0.5
	se_st=np.maximum(np.diag(cov).flatten(),1e-200)**0.5
	names=np.array(panel.args.names_v)
	
	tstat=np.maximum(np.minimum((args)/((se_robust<=0)*args*1e-15+se_robust),3000),-3000)
	tsign=1-scstats.t.cdf(np.abs(tstat),panel.df)
	sign_codes=get_sign_codes(tsign)
	
	return names,se_robust,se_st,tstat,tsign,sign_codes
	

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


def constraints_printout(constr,T):
	set_to,assco,cause=['']*T,['']*T,['']*T
	c=constr.fixed
	for i in c:
		set_to[i]=c[i].value_str
		assco[i]=c[i].assco_name
		cause[i]=c[i].cause	
		
	c=constr.intervals
	for i in c:
		if not c[i].intervalbound is None:
			set_to[i]=c[i].intervalbound
			assco[i]='NA'
			cause[i]=c[i].cause		
			
	return set_to,assco,cause
		


class output:
	def __init__(self,pr,panel,H,G,robustcov_lags,ll,constr,
	             sep='   ',startspace='',direction=None):
		args=ll.args_v		
		T=len(args)
		(names,se_robust,se_st,tstat,tsign,
		 sign_codes)=t_stats(panel, args, H, G,robustcov_lags)

		set_to,assco,cause=constraints_printout(constr,T)
		if direction is None:
			direction=T*['']		
		namelen=max([len(i) for i in names])
		for i in range(len(pr)):
			pr[i][0]=vars()[pr[i][0]]
			if type(pr[i][1])==str:
				pr[i][1]=vars()[pr[i][1]]

	
		output=[]
		output=np.concatenate([np.array(i[0]).reshape((T,1)) for i in pr],1)
		headings=[i[3] for i in pr]
		output=np.concatenate(([headings],output),0)
		
		self.printstring=coeficient_printout(pr,sep,startspace)
		
		self.outputmatrix=output
		self.names=names
		self.args=args
		self.se_robust=se_robust
		self.se_st=se_st
		self.tstat=tstat
		self.tsign=tsign
		self.sign_codes=sign_codes
		self.ll=ll
	
	def print(self,window=None):
		if window is None:
			print(self.printstring)
		else:
			window.update(self.printstring)
	
	def add_heading(self,Iterations=None,top_header=None,statistics=None):
		s=("LL: "+str(self.ll.LL)+'  ').ljust(23)
		if not Iterations is None:
			s+="Iteration: "+ str(Iterations).ljust(7)		
		if not statistics is None:
			for i in statistics:
				if not i[1] is None:
					if i[3]=='%':
						value=str(np.round(i[1]*100,i[2]))+i[3]
						s+=("%s: %s " %(i[0],value)).ljust(16)
					elif i[3]=='decimal': 
						value=np.round(i[1],i[2])
						s+=("%s: %s " %(i[0],value)).ljust(16)
					else:
						s+=str(i[0])+str(i[1])+str(i[3])
					
		s+='\n'
		if not top_header is None:
			s+=top_header+'\n'
		self.printstring=s+self.printstring
			
	def add_footer(self,text):
		self.printstring=self.printstring+'\n'+text

		
		
def coeficient_printout(pr, sep='   ',startspace=''):
	"""Prints output. pr is a list of a list for each statistic to be printed. The list for each statistic
	must have the format:\n
	[\n
	is not a string (boolean),
	variable, \n
	column width, \n
	name (string), \n
	correction for negative numbers (boolean),\n
	mid adjusted (True) or left adjusted (False)\n
	]"""

	T=len(pr[0][0])

	prstr=' '*4
	for i in range(len(pr)):#Headings
		a, l,notstr,name,neg,midjust=pr[i]
		if notstr:
			pr[i][0]=np.round(a,l-2).astype('<U'+str(l))
		prstr+=name.ljust(l)[:l]+sep
	prstr+='\n'
	for j in range(T):#Data:
		prstr+=startspace + (str(j)+'.').ljust(4)
		for i in range(len(pr)):
			a, l,notstr,name,neg,midjust=pr[i]
			v=str(a[j])
			if neg:
				if v[0]!='-':
					v=' '+v
			if midjust:
				prstr+=v.center(l)[:l]+sep
			else:
				prstr+=v.ljust(l)[:l]+sep
		prstr+='\n'

	return prstr


def sandwich(H,G,lags=3,ret_hessin=False):
	H=H*1
	sel=[i for i in range(len(H))]
	H[sel,sel]=H[sel,sel]+(H[sel,sel]==0)*1e-15
	hessin=np.linalg.inv(-H)
	V=stat.newey_west_wghts(lags,XErr=G)
	hessinV=cf.dot(hessin,V)
	sandw=cf.dot(hessinV,hessin)
	if ret_hessin:
		return sandw,hessin
	return sandw