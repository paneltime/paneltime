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
from matplotlib import pyplot  as plt

if 'win' in sys.platform:
	slash='\\'
else:
	slash='/'

class diagnostics:
	def __init__(self,panel,g,G,H,robustcov_lags,ll,simple_diagnostics=False):
		"""This class calculates, stores and prints statistics and diagnostics"""
		self.panel=panel
		ll.standardize(panel)
		self.savedir=get_savedir()
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(panel,ll)
		
		if simple_diagnostics:
			self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(10)
			self.norm_prob=stat.JB_normality_test(panel.e_st,panel.df)			
			return
		self.reg_output,names,args,se,tstat,tsign,sign_codes=self.coeficient_output(H,G,robustcov_lags,ll)
		self.coeficient_printout(names,args,se,tstat,tsign,sign_codes)
		
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(panel,ll,10)
		self.norm_prob=stat.JB_normality_test(ll.e_st,panel.df)		

		self.multicollinearity_check(G)

		self.data_correlations=self.correl()
		
		scatterplots(panel,self.savedir)

		print ( 'LL: %s' %(ll.LL,))
	
		self.adf_test=stat.adf_test(panel,ll,10)
		self.get_var_stats()
		self.save_stats(ll)
	
	def correl(self):
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
		c=np.round(c,3)
		c=np.concatenate((x_names,c),0)
		x_names=np.append([[0]],x_names,1).T
		c=np.concatenate((x_names,c),1)
		return c
		
		
		
	
	
	def coeficient_output(self,H,G,robustcov_lags,ll):
		panel=self.panel
		args=ll.args_v
		robust_cov_matrix=rp.sandwich(H,G,robustcov_lags)
		se=np.diag(robust_cov_matrix).flatten()**0.5
		names=panel.name_vector

		T=len(se)
		output=[]
		
		
		frstrw=['Dependent:',panel.y_name]
		frstrw.extend(['Rsq: ',str(self.Rsq),'Adjusted Rsq: ',str(self.Rsqadj),'LL-ratio: ',str(self.LL_ratio)])
		output.append(frstrw)		

		output.append(['Regressors:',names])
		output.append(['coef:',args])
		output.append(['SE:',se])
		tstat=np.maximum(np.minimum((args)/((se<=0)*args*1e-15+se),3000),-3000)
		output.append(['t-value:',tstat])	
		tsign=1-scstats.t.cdf(np.abs(tstat),panel.df)
		output.append(['t-sign:',tsign])
		sign_codes=get_sign_codes(tsign)
		output.append(['sign codes:',sign_codes])
		
		output=fix_savelist(output,True)
		
		return output,names,args,se,tstat,tsign,sign_codes
	
	def coeficient_printout(self,names,args,se,tstat,tsign,sign_codes):
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
		tsign=np.round(tsign,rndlen).astype('<U'+str(rndlen))
		sep='   '
		prstr='Variable names'.ljust(maxlen)[:maxlen]+sep
		prstr+='Coef'.ljust(rndlen)[:rndlen]+sep
		prstr+='SE'.ljust(rndlen)[:rndlen]+sep
		prstr+='t-stat.'.ljust(rndlen)[:rndlen]+sep
		prstr+='sign.'.ljust(rndlen)[:rndlen]+sep
		prstr+='\r'
		for i in range(T):
			b=str(args[i])
			t=str(tstat[i])
			if b[0]!='-':
				b=' '+b
				t=' '+t
			prstr+=names[i].ljust(maxlen)[:maxlen]+sep
			prstr+=b.ljust(rndlen)[:rndlen]+sep
			prstr+=se[i].ljust(rndlen)[:rndlen]+sep
			prstr+=t.ljust(rndlen)[:rndlen]+sep
			prstr+=tsign[i].ljust(rndlen)[:rndlen]+sep
			prstr+=sign_codes[i]
			prstr+='\r'
		prstr+='\r'+"Significance codes: .=0.1, *=0.05, **=0.01, ***=0.001"
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

	def get_var_stats(self):
		"""Assigns some statistics on the variabels to self"""
		panel=self.panel
		X=panel.X
		N,T,k=X.shape
		X_dev=stat.deviation(panel,X)
		avgs=stat.avg(panel,X)
		std=stat.std(panel,X_dev)
		vstat=np.concatenate((avgs,std,panel.xmin,panel.xmax))
		vstat=np.array(vstat,dtype='<U100')
		x_names=np.array(panel.x_names).reshape(1,k)
		self.var_stats=np.concatenate((x_names,vstat))



	def save_stats(self,ll,strappend=''):
		"""Saves the various statistics assigned to self"""
		panel=self.panel
		N,T,k=panel.X.shape
		save_list=[]
		save_list.append(['Type:',panel.descr])
		save_list.append(['LL:',ll.LL])
		save_list.append(['Number of groups:',N])
		save_list.append(['Maximum number of dates:',T])
		save_list.append(['Number of dates in each group:',panel.T_arr.reshape(N)])
		save_list.append(['(A) Total number of observations:',panel.NT])
		save_list.append(['(B) Observations lost to GARCH/ARIMA',panel.tot_lost_obs])		
		save_list.append(['    Total after loss of observations (A-B):',panel.NT_afterloss])
		save_list.append(['(C) Number of Random Effects coefficients:',N])
		save_list.append(['(D) Number of Fixed Effects coefficients in the variance process:',N])
		save_list.append(['(E) Number of coefficients:',panel.len_args])
		save_list.append(['DF (A-B-C-D-E):',panel.df])

		save_list.append(['RSq:',self.Rsq])
		save_list.append(['RSq Adj:',self.Rsqadj])
		save_list.append(['LL-ratio:',self.LL_ratio])
		save_list.append(['no ac_prob:',self.no_ac_prob])
		save_list.append(['norm prob:',self.norm_prob])
		save_list.append(['ADF (dicky fuller):',self.adf_test, "1% and 5 % lower limit of confidence intervals, respectively"])
		save_list.append([''])
		save_list.append(['Regression:',self.reg_output])
		
		save_list.append([''])
		save_list.append(['Correlation Matrix:'])
		save_list.extend(self.data_correlations)
		save_list.append([''])
		save_list.append([''])
		save_list.append(['Multicollinearity test - condition matrix:'])		
		save_list.extend(self.MultiColl)
		save_list.append([''])
		save_list.append([''])
		save_list.append(['Variable statistics:'])
		save_list.append(['',self.var_stats[0]])
		save_list.append(['Means:',self.var_stats[1]])
		save_list.append(['SEs:',self.var_stats[2]])
		save_list.append(['Min:',self.var_stats[3]])
		save_list.append(['Max:',self.var_stats[4]])
		
		save_list=fix_savelist(save_list)


		write_csv_matrix_file(panel.descr+strappend,save_list,self.savedir)
		
		pass
	
def fix_savelist(save_list,transpose=False):
	new_savelist=[]
	for i in range(len(save_list)):
		d=get_list_dim(save_list[i])
		if d==0:
			new_savelist.append([save_list[i]])
		elif d==1:
			lst=[]
			for j in range(len(save_list[i])):
				s=save_list[i][j]
				d=get_list_dim(s)
				if d==0:
					lst.append(s)
				elif d==1:
					lst.extend(s)
				else:
					new_savelist.append(lst)
					lst=[]
					for k in s:
						new_savelist.append(k)
					break
			new_savelist.append(lst)
		elif d==2:
			for k in save_list[i]:
				new_savelist.append(k)
	if transpose:#requires a matrix
		n=len(new_savelist)
		k=len(new_savelist[0])
		t_savelist=[n*[None] for i in range(k)]
		for i in range(n):
			for j in range(k):
				t_savelist[j][i]=new_savelist[i][j]
		new_savelist=t_savelist
	return new_savelist

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


def write_csv_matrix_file(filename,Variable,savedir):
	"""stores the contents of Variable to a file named filename"""
	filename=filename.replace('.csv','')+'.csv'
	savedir=savedir +slash+ filename
	print ( 'saves to %s' %(savedir,))
	file = open(savedir,'w',newline='')
	writer = csv.writer(file,delimiter=';')
	writer.writerows(Variable)
	file.close()
	
def get_savedir():
	savedir=os.getcwd()+slash+'output'
	if not os.path.exists(savedir):
		os.makedirs(savedir)
	return savedir
	
	
def scatterplots(panel,savedir):
	
	x_names=panel.x_names
	y_name=panel.y_name
	X=panel.raw_X
	Y=panel.raw_Y
	N,k=X.shape
	for i in range(k):
		plt.cla()
		plt.scatter(X[:,i],Y[:,0], alpha=.1, s=10)
		plt.ylabel(y_name)
		plt.xlabel(x_names[i])
		fgr=plt.figure(0)
		xname=remove_illegal_signs(x_names[i])
		fgr.savefig(savedir+slash+'%s-%s.png' %(y_name,xname))
	
	
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
