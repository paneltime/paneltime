#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file



import numpy as np
import stat_functions as stat
from scipy import stats as scstats
import functions as fu
import loglikelihood as logl
STANDARD_LENGTH=10

class statistics:
	def __init__(self,results,correl_vars=None,descriptives_vars=None,simple_statistics=False):
		"""This class calculates, stores and prints statistics and statistics"""		

		self.G=results.direction.G
		self.H=results.direction.H
		self.ll=results.ll
		self.panel=results.panel
		self.ll.standardize()
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(self.panel,self.ll)
		self.LL_restricted=logl.LL(self.panel.args.args_restricted, self.panel).LL
		self.LL_OLS=logl.LL(self.panel.args.args_OLS, self.panel).LL		
		
		
		
		if simple_statistics:	
			self.output=self.arrange_output(results.direction)
			return	
		
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(self.panel,self.ll,10)
		self.norm_prob=stat.JB_normality_test(self.ll.e_st,self.panel)		
		self.output=self.arrange_output(results.direction,self.norm_prob,self.no_ac_prob)
		self.reg_output=self.output.outputmatrix
		self.multicollinearity_check(self.G)

		self.data_correlations,self.data_statistics=self.correl_and_statistics(correl_vars,descriptives_vars)
		
		self.adf_test=stat.adf_test(self.panel,self.ll,10)
		self.save_stats()
		
	def arrange_output(self,direction,norm_prob,ac_prob):
		panel,H,G,ll=self.panel,self.H,self.G,self.ll
		l=STANDARD_LENGTH
		# python variable name,	lenght ,	not string,		display name,			can take negative values,	justification	col space
		pr=[
			['names',			'namelen',	False,			'Variable names',		False,						'right', 		''],
	        ['args',			l,			True,			'Coef',					True,						'right', 		'  '],
	        ['se_robust',		l,			True,			'robust',				False,						'right', 		'  '],
	        ['se_st',			l,			True,			'standard',				False,						'right', 		'  '],
	        ['tstat',			l,			True,			't-stat.',				True,						'right', 		'  '],
	        ['tsign',			l,			True,			'sign.',				False,						'right', 		'  '],
	        ['sign_codes',		4,			False,			'',						False,						'left', 		'  '],
		    ['dx_norm',		    l,			True,			'direction',	    	True,						'right', 		'  '],
			['multicoll',		1,			False,			'',						False,						'right', 		''],
			['assco',			20,			False,			'associated variable',	False,						'center', 		'  '],
		    ['set_to',			6,			False,			'restr',			False,						'center', 		'  '],
		    ['cause',			l,			False,			'cause',				False,						'right', 		'  ']
		]
		o=output(pr, ll,direction,panel.settings.robustcov_lags_statistics.value)
		o.add_heading(top_header=' '*100 + '_'*11+'SE'+'_'*11+" "*73+"_"+"restricted variables"+"_",
		              statistics= [['Normality (Jarque-Bera test for normality)',norm_prob,3,'%'], 
		                           ['Stationarity (Breusch Godfrey_test on AC, significance)',ac_prob,3,'%']])
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
		    ['Description:',panel.input.descr],
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
		    ['Dependent:',panel.input.y_name]
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
		
		fname=panel.input.descr.replace('\n','').replace('\r', '')
		if len(fname)>65:
			fname=fname[:30]+'...'+fname[-30:]
		fu.savevar(output_table,fname+'.csv')
		
		self.output_dict=output
		
def t_stats(args,direction,lags):
	names=np.array(direction.panel.args.names_v)
	T=len(names)
	if direction.H is None:
		return names,T*[np.nan],T*[np.nan],T*[np.nan],T*[np.nan],T*[np.nan]
	se_robust,se_st=sandwich(direction,lags)
	no_nan=np.isnan(se_robust)==False
	valid=no_nan
	valid[no_nan]=(se_robust[no_nan]>0)
	tstat=np.array(T*[np.nan])
	tsign=np.array(T*[np.nan])
	tstat[valid]=args[valid]/se_robust[valid]
	tsign[valid]=1-scstats.t.cdf(np.abs(tstat[valid]),direction.panel.df)
	sign_codes=get_sign_codes(tsign)
	
	return names,se_robust,se_st,tstat,tsign,sign_codes
	

def add_variable(name,panel,names,variables):
	if '|~|' in name:
		return
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
		if np.isnan(i):
			sc.append(i)
		elif i<0.001:
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


def constraints_printout(direction,T):
	panel=direction.panel
	constr=direction.constr
	collinears=direction.collinears
	dx_norm=direction.dx_norm
	if dx_norm is None:
		dx_norm=[0]*T
	set_to,assco,cause,multicoll=['']*T,['']*T,['']*T,['']*T
	if constr is None:
		return set_to,assco,cause,multicoll,dx_norm
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
			
	for i in collinears:#adding associates of non-severe multicollinearity
		multicoll[i]='|'
		if i not in set_to:
			assco[i]=panel.args.names_v[collinears[i][0]]		
	return set_to,assco,cause,multicoll,dx_norm


class output:
	def __init__(self,pr,ll,direction,lags,startspace=''):
		self.ll=ll
		args=self.ll.args_v		
		T=len(args)
		(names,se_robust,se_st,tstat,tsign,
		 sign_codes)=t_stats(args,direction,lags)

		set_to,assco,cause,multicoll,dx_norm=constraints_printout(direction,T)	
		
		insert_variables(pr, names, se_robust, se_st, tstat, tsign, sign_codes, 
							set_to, assco, cause, multicoll, dx_norm,args)		
		output=[]
		output=np.concatenate([np.array(i[0]).reshape((T,1)) for i in pr],1)
		headings=[i[3] for i in pr]
		output=np.concatenate(([headings],output),0)
		
		self.printstring=coeficient_printout(pr,startspace)
		
		self.outputmatrix=output
		self.names=names
		self.args=args
		self.se_robust=se_robust
		self.se_st=se_st
		self.tstat=tstat
		self.tsign=tsign
		self.sign_codes=sign_codes

	
	def print(self,window=None):
		if window is None:
			print(self.printstring)
		else:
			window.gui_main_tabs.replace_all(self.printstring)
	
	def add_heading(self,Iterations=None,top_header=None,statistics=None,incr=None):
		s=("LL: "+str(self.ll.LL)+'  ').ljust(23)
		if not incr is None:
			s+=("Increment: "+ str(incr)).ljust(17)+"  "
		else:
			s+=str(" ").ljust(19)
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

		
def insert_variables(pr,names,se_robust,se_st,tstat,tsign,
		 sign_codes,set_to,assco,cause,multicoll,dx_norm,args):
	namelen=max([len(i) for i in names])
	for i in range(len(pr)):
		pr[i][0]=vars()[pr[i][0]]
		if type(pr[i][1])==str:
			pr[i][1]=vars()[pr[i][1]]	
		
def coeficient_printout(pr, startspace=''):
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

	prstr=' '*6
	for i in range(len(pr)):#Headings
		a, l,notstr,name,neg,just,sep=pr[i]
		if notstr:
			pr[i][0]=round(a,l)
		prstr+=sep+name.ljust(l)[:l]
	prstr+='\n'
	for j in range(T):#Data:
		prstr+=startspace + (str(j)+'.').ljust(4)
		for i in range(len(pr)):
			a, l,notstr,name,neg,just,sep=pr[i]
			prstr+=sep
			v=str(a[j])
			if neg:
				if v[0]!='-':
					v=' '+v
			if just=='center':
				prstr+=v.center(l)[:l]
			elif just=='right':
				prstr+=v.rjust(l)[:l]
			else:
				prstr+=v.ljust(l)[:l]
		prstr+='\n'

	return prstr

def round(a,l):
	return np.round(a,l-2).astype('<U'+str(l))

def sandwich(direction,lags):
	panel=direction.panel
	H,G,delmap,idx=reduce_size(direction)
	lags=lags+panel.lost_obs
	hessin=np.linalg.inv(-H)
	se_robust,se=stat.robust_se(panel,lags,hessin,G)
	se_robust,se=expand_x(se_robust, idx),expand_x(se, idx)
	return se_robust,se

def reduce_size(direction):
	H=direction.H
	G=direction.G
	remove_dict=direction.collinears
	m=len(H)
	idx=np.ones(m,dtype=bool)
	delmap=np.arange(m)
	if len(list(remove_dict.keys()))>0:#removing fixed constraints from the matrix
		idx[list(remove_dict.keys())]=False
		H=H[idx][:,idx]
		G=G[:,:,idx]
		delmap-=np.cumsum(idx==False)
		delmap[idx==False]=m#if for some odd reason, the deleted variables are referenced later, an out-of-bounds error is thrown	
	return H,G,delmap,idx

def expand_x(x,idx):
	m=len(idx)
	x_full=np.zeros(m)
	x_full[:]=np.nan
	x_full[idx]=x
	return x_full
	
	
	