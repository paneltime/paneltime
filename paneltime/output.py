#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file

import numpy as np
from scipy import stats as scstats
from tkinter import font as tkfont
import stat_functions as stat
STANDARD_LENGTH=8
		


class output:
	def __init__(self,ll,direction,lags,include_cols,incr,iterations):
		self.ll=ll
		self.direction=direction
		self.panel=self.ll.panel
		self.n_variables=self.panel.args.n_args
		self.n_rows=2*(self.n_variables+1)
		self.n_cols=len(include_cols)
		self.d={'args':self.ll.args_v,'count':range(self.n_variables)}		
		t_stats(self.d['args'],direction,lags,self.d)
		constraints_printout(direction,self.d)
		self.incr=incr
		self.include_cols=include_cols
		self.iterations=iterations

	def table(self):
		return table_obj(self, self.d)
		
	def statistics(self):
		panel=self.panel
		ll=self.ll
		ll.standardize()
		N,T,k=panel.X.shape
		Rsq, Rsqadj, LL_ratio,LL_ratio_OLS=stat.goodness_of_fit(panel,ll)	
		no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(panel,ll,10)
		norm_prob=stat.JB_normality_test(ll.e_st,panel)
		
		#Description:{self.panel.input.desc}
		s=statistics(panel,N,T,Rsq,Rsqadj,LL_ratio,no_ac_prob,norm_prob)
		adf=adf_str(panel, ll)
		return s+adf
	
	def heading(self):
		statistics=[['\nDependent: ',self.panel.input.y_name[0],None,"\n"],
					['Max condition index',self.direction.CI,3,'decimal']]
		
		s=("LL: "+str(self.ll.LL)+'  ').ljust(23)
		if not self.incr is None:
			s+=("Increment: "+ str(self.incr)).ljust(17)+"  "
		else:
			s+=str(" ").ljust(19)
		if not self.iterations is None:
			s+="Iteration: "+ str(self.iterations).ljust(7)
			
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
		return s		



	
class table_obj(dict):
	def __init__(self,output,d):
		dict.__init__(self)
		self.columns=dict()
		for a, l,is_string,name,neg,just,sep,default_digits in pr:		
			self[a]=column(d,a,l, is_string, name, neg, just, sep, default_digits,output.n_variables)
		self.n_cols=output.n_cols
		self.include_cols=output.include_cols
		self.n_variables=output.n_variables
		self.n_rows=output.n_rows
		self.heading=output.heading()
		self.footer=f"\nSignificance codes: '=0.1, *=0.05, **=0.01, ***=0.001,    |=collinear\n\n{output.ll.err_msg}"
		self.statistics=output.statistics()
		
			
	def table(self,digits):
		self.X=self.output_matrix(digits)	
		if not self.constr_pos is None:
			p="\t"*self.constr_pos+"constraints".center(38)+"\n"
		for i in range(len(self.X)):
			for j in range(len(self.X[0])):
				p+='\t'+self.X[i][j]
			p+='\n'
		return p
		
	def get_tab_stops(self,font,init_space=1):
		font = tkfont.Font(font=font)
		m_len = font.measure("m")
		counter=init_space*m_len
		tabs=[f"{counter}"]
		for i in range(self.n_cols):
			a=self.include_cols[i]
			if type(a)==list:
				a=a[0]
			spc=self[a].tab_sep
			t=font.measure(self.X[0][i])+spc*m_len
			counter+=t
			tabs.extend([f"{counter}"])			
		return tabs
	
	def output_matrix(self,digits):
		X=[['']*self.n_cols for i in range(self.n_rows)]
		for i in range(self.n_cols):
			a=self.include_cols[i]
			if type(a)==list:
				X[0][i]=self[a[0]].name
				X[1][i]=f"[{self[a[1]].name}]"
				v=[self[a[j]].values(digits) for j in range(3)]
				for j in range(self.n_variables):
					X[(j+1)*2][i]=v[0][j]
					X[(j+1)*2+1][i]=f"[{v[1][j]}]{v[2][j]}"
			else:
				X[0][i]=self[a].name
				v=self[a].values(digits)
				for j in range(self.n_variables):
					X[(j+1)*2][i]=v[j]
		self.format_X(X)
		return X	

	def format_X(self,X):
		x=0
		self.constr_pos=None
		for i in range(self.n_cols):
			a=self.include_cols[i]
			if a=='assco':
				self.constr_pos=i	
			if type(a)==list:
				a=a[0]		
			if self[a].is_string:
				x+=self.format_X_col_str(X,i,a)
			else:
				x+=self.format_X_col_num(X,i)
	
	def format_X_col_str(self,X,i,a):
		l_tot=0
		if self[a].length=='namelen':
			for j in range(self.n_rows):
				l_tot=max((len(X[j][i]),l_tot))
		else:
			l_tot=self[a].length
		X[0][i]=X[0][i].ljust(l_tot)
		X[1][i]=X[1][i].center(l_tot)		
		for j in range(2,self.n_rows):
			if self[a].justification=='right':
				X[1][i].rjust(l_tot)
			elif self[a].justification=='left':
				X[1][i].ljust(l_tot)
			else:
				X[1][i].center(l_tot)
		return l_tot

	def format_X_col_num(self,X,i):
		l_max=[0,0]
		s=[[None,None] for i in range(self.n_rows)]
		for j in range(2,self.n_rows):
			if '.' in X[j][i]:
				s[j]=X[j][i].split('.')
				l_max=max((l_max[0],len(s[j][0]))),max((l_max[1],len(s[j][1])))
		l_sum=sum(l_max)
		l_tot=max((l_sum+1,len(X[0][i]),len(X[1][i])))
		l_extra=l_tot-l_sum-1
		X[0][i]=X[0][i].ljust(l_tot)
		X[1][i]=X[1][i].center(l_tot)
		for j in range(2,self.n_rows):
			if '.' in X[j][i]:
				left=s[j][0].rjust(l_extra+l_max[0])
				right=s[j][1].ljust(l_max[1])
				X[j][i]=(left+'.'+right)
			else:
				X[j][i]=X[j][i].center(l_tot)
		return l_tot	
	

	
class column:
	def __init__(self,d,a,l,is_string,name,neg,just,sep,default_digits,n_variables):		
		self.length=l
		self.is_string=is_string
		self.name=name
		self.default_digits=default_digits
		self.neg_allowed=neg
		self.justification=just
		self.tab_sep=sep
		self.n_variables=n_variables
		if a in d:
			self.exists=True
			self.input=d[a]
		else:
			self.exists=False
			self.input=[' - ']*self.n_variables		
		
	def values(self,digits):
		l=self.length
		vals=self.input
		if not self.exists:
			return vals
		v=np.array(vals).astype('<U128')
		if self.is_string:
			return v
		v_rnd=vals
		if not self.default_digits is None:
			if self.default_digits==-1:
				if not digits=='':
					v_rnd=np.round(vals,digits)	
			else:
				v_rnd=np.round(vals,self.default_digits)	
		rnd_scientific=True
		for i in range(len(v)):
			if 'e' in v[i] and not rnd_scientific:
				v[i]=np.format_float_scientific(vals[i],l-7)
			else:
				if 'e' in str(v_rnd[i]):
					v[i]=np.format_float_scientific(vals[i],l-7)
				else:
					v[i]=str(v_rnd[i])[:l]
		return v			
	
			

	
def get_preferences(output_gui):
	try:
		pref=output_gui.window.right_tabs.preferences.options
		return pref
	except:
		return
	
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
	if (G is None) or (H is None):
		return
	weak_mc_dict=direction.weak_mc_dict.keys()
	constr=list(direction.constr.fixed.keys())
	for i in weak_mc_dict:
		if not i in constr:
			constr.append(i)
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
			sc.append("'  ")
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
		
	
def t_stats(args,direction,lags,d):
	d['names']=np.array(direction.panel.args.names_v)
	T=len(d['names'])
	if direction.H is None:
		return
	d['se_robust'],d['se_st']=sandwich(direction,lags)
	no_nan=np.isnan(d['se_robust'])==False
	valid=no_nan
	valid[no_nan]=(d['se_robust'][no_nan]>0)
	d['tstat']=np.array(T*[np.nan])
	d['tsign']=np.array(T*[np.nan])
	d['tstat'][valid]=args[valid]/d['se_robust'][valid]
	d['tsign'][valid]=(1-scstats.t.cdf(np.abs(d['tstat'][valid]),direction.panel.df))#Two sided tests
	d['sign_codes']=get_sign_codes(d['tsign'])


def adf_str(panel,ll):
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
		return adf
	
def statistics(panel,N,T,Rsq,Rsqadj,LL_ratio,no_ac_prob,norm_prob):
		return f"""
\t                         
\tNumber of IDs                       :\t{N}
\tNumber of dates (maximum)           :\t{T}

\tA) Total number of observations     :\t{panel.NT_before_loss}
\tB) Observations lost to GARCH/ARIMA':\t{panel.tot_lost_obs}		
\tRandom/Fixed Effects in
\tC)      Mean process                :\t{panel.number_of_RE_coef}
\tD)      Variance process            :\t{panel.number_of_RE_coef_in_variance}
\tE) Number of coefficients in regr.  :\t{panel.args.n_args}
\tDegrees of freedom (A-B-C-D-E)      :\t{panel.df}

\tR-squared                           :\t{round(Rsq*100,1)}%
\tAdjusted R-squared                  :\t{round(Rsqadj*100,1)}%
\tLL-ratio                            :\t{round(LL_ratio,2)}
\tBreusch-Godfrey-test                :\t{round(no_ac_prob*100,1)}% (significance, probability of no auto correlation)
\tJarque–Bera test for normality      :\t{round(norm_prob*100,1)}% (significance, probability of normality)
"""	

l=STANDARD_LENGTH
#python variable name,	length,		is string,  display name,		neg. values,	justification	next tab space		round digits (None=no rounding,-1=set by user)
pr=[
		['count',		2,			False,		'',					False,			'right', 		2,					None],
		['names',		'namelen',	True,		'Variable names',	False,			'right', 		2, 					None],
		['args',		l,			False,		'Coef',				True,			'right', 		2, 					-1],
		['se_robust',	l,			False,		'robust SE',		True,			'right', 		3, 					-1],
		['sign_codes',	5,			True,		'',					False,			'left', 		2, 					-1],
		['dx_norm',	l,				False,		'direction',		True,			'right', 		2, 					None],
		['tstat',		l,			False,		't-stat.',			True,			'right', 		2, 					2],
		['tsign',		l,			False,		'p-value',			False,			'right', 		2, 					3],
		['multicoll',	1,			True,		'',					False,			'left', 		2, 					None],
		['assco',		20,			True,		'collinear with',	False,			'center', 		2, 					None],
		['set_to',		6,			True,		'set to',			False,			'center', 		2, 					None],
		['cause',		50,			True,		'cause',			False,			'right', 		2, 					None]]		