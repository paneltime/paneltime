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
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(self.panel,self.ll)
		self.LL_restricted=logl.LL(self.panel.args.args_restricted, self.panel).LL
		self.LL_OLS=logl.LL(self.panel.args.args_OLS, self.panel).LL		
		self.name=name
		
		self.no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(self.panel,self.ll,10)
		self.norm_prob=stat.JB_normality_test(self.ll.e_st,self.panel)	
		if simple_statistics:	
			self.output=self.arrange_output(results.direction,self.norm_prob,self.no_ac_prob)
			return	
		
		self.output=self.arrange_output(results.direction,self.norm_prob,self.no_ac_prob)
		self.reg_output=self.output.X
		self.multicollinearity_check(self.G)

		self.data_correlations,self.data_statistics=self.correl_and_statistics(correl_vars,descriptives_vars)
		
		self.adf_test=stat.adf_test(self.panel,self.ll,10)
		self.save_stats()
		
	def arrange_output(self,direction,norm_prob,ac_prob):
		panel,H,G,ll=self.panel,self.H,self.G,self.ll
		l=STANDARD_LENGTH
		# python variable name,	lenght ,	is string,		display name,			can take negative values,	justification	next tab space
		pr=[
			['names',			'namelen',	True,			'Variable names',		False,						'right', 		2],
	        ['args',			l,			False,			'Coef',					True,						'right', 		2],
	        ['se_robust',		l,			False,			'robust',				False,						'right', 		2],
	        ['se_st',			l,			False,			'standard',				False,						'right', 		2],
	        ['tstat',			l,			False,			't-stat.',				True,						'right', 		2],
	        ['tsign',			l,			False,			'sign.',				False,						'right', 		1],
	        ['sign_codes',		4,			True,			'',						False,						'left', 		2],
		    ['dx_norm',		    l,			False,			'direction',	    	True,						'right', 		2],
			['multicoll',		1,			True,			'',						False,						'right', 		2],
			['assco',			20,			True,			'associated variable',	False,						'center', 		2],
		    ['set_to',			6,			True,			'restr',			False,							'center', 		2],
		    ['cause',			10,			True,			'cause',				False,						'right', 		2]
		]
		o=output(pr, ll,direction,panel.settings.robustcov_lags_statistics.value[0])
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
		if self.name is None:
			fname=panel.input.descr.replace('\n','').replace('\r', '')
		else:
			fname=self.name
		if len(fname)>65:
			fname=fname[:30]+'...'+fname[-30:]
		fu.savevar(output_table,fname+'.csv')
		
		self.output_dict=output
	
def t_stats(args,direction,lags,d):
	d['names']=np.array(direction.panel.args.names_v)
	if direction.H is None:
		return
	d['se_robust'],d['se_st']=sandwich(direction,lags)
	no_nan=np.isnan(d['se_robust'])==False
	valid=no_nan
	valid[no_nan]=(d['se_robust'][no_nan]>0)
	T=len(d['names'])
	d['tstat']=np.array(T*[np.nan])
	d['tsign']=np.array(T*[np.nan])
	d['tstat'][valid]=args[valid]/d['se_robust'][valid]
	d['tsign'][valid]=1-scstats.t.cdf(np.abs(d['tstat'][valid]),direction.panel.df)
	d['sign_codes']=get_sign_codes(d['tsign'])

	

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

def constraints_printout(direction,d):
	panel=direction.panel
	constr=direction.constr
	weak_mc_dict=direction.weak_mc_dict
	if not direction.dx_norm is None:
		d['dx_norm']=direction.dx_norm
	T=len(d['names'])
	d['set_to'],d['assco'],d['cause'],d['multicoll']=['']*T,['']*T,['']*T,['']*T
	if constr is None:
		return
	c=constr.fixed
	for i in c:
		d['set_to'][i]=c[i].value_str
		d['assco'][i]=c[i].assco_name
		d['cause'][i]=c[i].cause	
		
	c=constr.intervals
	for i in c:
		if not c[i].intervalbound is None:
			d['set_to'][i]=c[i].intervalbound
			d['assco'][i]='NA'
			d['cause'][i]=c[i].cause		
			
	for i in weak_mc_dict:#adding associates of non-severe multicollinearity
		d['multicoll'][i]='|'
		d['assco'][i]=panel.args.names_v[weak_mc_dict[i][0]]		

class output:
	def __init__(self,pr,ll,direction,lags):
		self.ll=ll
		d={'args':self.ll.args_v}		
		t_stats(d['args'],direction,lags,d)
		constraints_printout(direction,d)	
		self.X,self.tab_spaces=output_matrix(pr, d)		
		p=""
		for i in range(len(self.X)):
			for j in range(len(self.X[0])):
				p+='\t'+self.X[i][j]
			p+='\n'
		self.printstring=p
		self.dict=d
		self.panel=ll.panel
		self.ll=ll
	
	def get_tab_stops(self,font,init_space=1):
		font = tkfont.Font(font=font)
		m_len = font.measure("m")
		counter=init_space*m_len
		tabs=[f"{counter}"]
		n=len(self.X[1])
		for i in range(n):
			counter+=len(self.X[1][i])*m_len
			counter+=self.tab_spaces[i]*m_len
			tabs.extend([f"{counter}"])
		return tabs
		
	def print(self,tab=None):
		if tab is None:
			print(self.printstring)
		else:
			tab.box.replace_all(self.printstring)
	
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
		self.printstring+='\n'+text
		if True:
			self.printstring+='\n\n'+self.statistics()
		
	def statistics(self):
		panel=self.panel
		ll=self.ll
		ll.standardize()
		N,T,k=panel.X.shape
		Rsq, Rsqadj, LL_ratio,LL_ratio_OLS=stat.goodness_of_fit(panel,ll)	
		no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(panel,ll,10)
		norm_prob=stat.JB_normality_test(ll.e_st,panel)

		#Description:{self.panel.input.desc}
		s=f"""
\t                         
\tNumber of IDs                       :{N}
\tNumber of dates (maximum)           :{T}

\tA) Total number of observations     :{panel.NT_before_loss}
\tB) Observations lost to GARCH/ARIMA':{panel.tot_lost_obs}		
\tRandom/Fixed Effects in
\tC)      Mean process                :{panel.number_of_RE_coef}
\tD)      Variance process            :{panel.number_of_RE_coef_in_variance}
\tE) Number of coefficients in regr.  :{panel.args.n_args}
\tDegrees of freedom (A-B-C-D-E)      :{panel.df}

\tR-squared                           :{round(Rsq*100,1)}%
\tAdjusted R-squared                  :{round(Rsqadj*100,1)}%
\tLL-ratio                            :{round(LL_ratio,2)}
\tBreusch-Godfrey-test                :{round(no_ac_prob*100,1)}% (significance, probability of no auto correlation)
\tJarque–Bera test for normality      :{round(norm_prob*100,1)}% (significance, probability of normality)
"""
		ADF_stat,c1,c5=stat.adf_test(panel,ll,10)
		if not ADF_stat=='NA':
			if ADF_stat<c1:
				ADF_res="Unit root rejected at 1%"
			elif ADF_stat<c5:
				ADF_res="Unit root rejected at 5%"
			else:
				ADF_res="Unit root not rejected"		
			adf=f"""
\tAugmented Dicky-Fuller (ADF)        
\t                   Test statistic   :{round(ADF_stat,2)}
\t                   1% critical value:{round(c1,2)}
\t                   5% critical value:{round(c5,2)}
\t                   Result           :{ADF_res}
			"""
		else:
			adf="Unable to calculate ADF"
		if panel.df<1:
			s+="""
WARNING: All your degrees of freedom (df) has been consumed, so statistics cannot be computed.
you can increase df by for example turning off random/fixed effects """
		return s+adf
		
def output_matrix(pr,d):
	"""Formats the output to an output matrix
	pr must have the format directions:\n
	[\n
	is not a string (boolean),
	variable, \n
	column width, \n
	name (string), \n
	correction for negative numbers (boolean),\n
	mid adjusted (True) or left adjusted (False)\n
	]"""	
	T=len(d['names'])
	K=len(pr)
	X=[['   ']+[str(i).rjust(2) for i in range(T)]]
	tab_spaces=[2]
	for a, l,is_string,name,neg,just,sep in pr:
		if a in d:
			if a=='names':
				l=max([len(i) for i in d['names']])
			v=format_var(d[a], l,is_string,name,neg,just,sep)
			X.append([name[:l+sep-1].ljust(l+sep-1)]+list(v))
			tab_spaces.append(sep)
	ret=[[None]*len(X) for i in range(len(X[0]))]
	for i in range(len(X)):#transposing
		for j in range(len(X[0])):
			ret[j][i]=X[i][j]
	return ret,tab_spaces

def format_var(vals,l,is_string,name,neg,just,sep):
	v=np.array(vals).astype('<U128')	
	if is_string:
		for i in range(len(v)):
			if just=='center':
				v[i]=v[i].center(l)[:l]
			elif just=='right':
				v[i]=v[i].rjust(l)[:l]
			else:
				v[i]=v[i].ljust(l)[:l]	
	else:
		max_l=0
		a=[None]*len(v)
		for i in range(len(v)):
			if 'e' in v[i]:
				v[i]=np.format_float_scientific(vals[i],l-7)
			a[i]=v[i].split('.')
			max_l=max((max_l,len(a[i][0])))			
		if max_l>l-3:
			for i in range(len(v)):
				v[i]=np.format_float_scientific(vals[i],l-3)
			return v
		for i in range(len(v)):
			if len(a[i])==2:
				v[i]=(a[i][0].rjust(max_l) + '.' + a[i][1])[:l].ljust(l)
			else:
				v[i]=v[i][:l].center(l)

	
	return v
	

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
	weak_mc_dict=direction.weak_mc_dict.keys()
	constr=list(direction.constr.fixed.keys())
	for i in weak_mc_dict:
		if not i in constr:
			constr.append[i]
	m=len(H)
	idx=np.ones(m,dtype=bool)
	delmap=np.arange(m)
	if len(constr)>0:#removing fixed constraints from the matrix
		idx[constr]=False
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
	
	
	