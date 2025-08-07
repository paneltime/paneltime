#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file

from . import stat_functions as stat
from . import stat_dist
from ..processing import arguments
import textwrap

import numpy as np
import time
from datetime import datetime

STANDARD_LENGTH=8

INITVAR = arguments.INITVAR


class Output:
	#This class handles auxilliary statistics (regression info, diagnostics, df accounting)
	def __init__(self,comm,panel):

		self.ll=comm.ll
		self.panel=panel
		self.delta_time = 0
		self.incr=0
		self.dx_norm = comm.dx_norm

		
		self.define_table_size()
		self.statistics_decimals = 3 
		self.update(comm, 0)
		self.describe()

	def describe(self):
		s =  f"{self.panel.input.Y_names[0]} = {' + '.join(self.panel.input.X_names)}"
		s = textwrap.TextWrapper(width=self.stat_totlen).fill(s) + '\n'   
		if len(self.panel.input.Z_names[1:]):
			s += f"Instruments:\t{', '.join(self.panel.input.Z_names[1:])}\n"     
		s = "Regression model:\n" + s 
		self.model_desc = s

	def update(self,comm, delta_time):
		self.iterations = comm.its
		self.ll = comm.ll
		self.dx_norm = comm.dx_norm
		self.incr = comm.incr
		self.delta_time = delta_time
		self.conv = comm.conv
		self.msg = comm.msg
		self.constr = comm.constr
		self.stats = Statistics(comm,self.panel, delta_time) 

		
	def statistics(self):
		s = self.stats
		heading = 'Statistics:'

		# Construct c0 and c1 tuples using the assigned variables
		c0 = (
			('Dep. Variable:', s.info.dep_var), 
			('Model:', s.info.model), 
			('Method:', s.info.method), 
			('Date:', s.info.date_str), 
			('Time:', s.info.time_str), 
			('Run time (its) [conv]:', s.info.run_time_str), 
			('Observations count:', s.info.obs_count), 
			('Df Residuals:', s.info.df), 
			('Df Model:', s.info.df_model), 
			('Covariance Type:', s.info.cov_type)
		)

		c1 = (
			('R-squared:', s.diag.Rsq), 
			('Adj R-squared:', s.diag.Rsqadj), 
			('F-statistic:', s.diag.F), 
			('Prob (F-statistic):', s.diag.F_p), 
			('Log-Likelihood:', s.info.log_lik), 
			('AIC:', s.info.aic), 
			('BIC:', s.info.bic), 
			('Panel groups:', s.info.panel_groups), 
			('Panel dates:', s.info.panel_dates), 
			('Initial variance:', s.info.initvar)
		)

		tbl = [(c0[i],c1[i]) for i in range(len(c0))]
		return heading + '\n' + self.parse_tbl(tbl)  
	
	
	def diagnostics(self):

		s = self.stats
		heading = 'Diagnostics:'
		c0 =(	('Distribution:',''),
				('  Omnibus:',s.diag.Omnibus_st),
				('  Prob(Omnibus):',s.diag.Omnibus_pval),
				('  Jarque-Bera (JB):',s.diag.JB_st),
				('  Prob(JB):',s.diag.JB_prob_st),
				('  Skew:',s.diag.skewness_st),
				('  Kurtosis:',s.diag.kurtosis),
				('','')
				
		)   
		
		c1 =(	('Stationarity:',''),
				('  Durbin-Watson:',s.diag.DW),
				('  ADF statistic:',s.diag.ADF_stat),
				('  ADF crit.val 1%:',s.diag.c1),
				('  ADF crit.val 5%:',s.diag.c5),
				('Singularity:',''),
				('  Cond. No.:',s.diag.ci),
				('  Cond. var count.:', s.diag.n_ci),

		)     
		
		tbl = [(c0[i],c1[i]) for i in range(len(c0))]
		return heading + '\n' + self.parse_tbl(tbl)  
	
	def df_accounting(self):

		N,T,k=self.panel.X.shape
		heading = 'Df accounting:'
		samsum = [
			('SAMPLE SIZE SUMMARY:',''),
			('Original sample size:',self.panel.orig_size),
			('Sample size after removals:',self.panel.NT_before_loss),
			('Degrees of freedom:',self.panel.df),
			('Number of variables:',k),
			('Number of groups:',N),
			('Number of dates (maximum):',T)
		]    
		
		grpremv = [
				 ('GROUP REMOVAL:',''),
				 ('Lost observations:',''), 
				 ('A) ARIMA/GARCH:', self.panel.lost_obs), 
				 ('B) min obs (user preferences):',self.panel.options.min_group_df),
				 ('Required obs (A+B):',self.panel.lost_obs+self.panel.options.min_group_df),
				 
				 ('Groups removed:',''),
				 ('A) total # of groups:',len(self.panel.idincl)),
				 ('B) # of groups removed:',sum(self.panel.idincl==False)), 
				 ('# of groups remaining (A-B):',sum(self.panel.idincl==True)), 
				 ('# of observations removed:',self.panel.orig_size-self.panel.NT_before_loss)
				 ]    
		

		
		df = [('DEGREES OF FREEDOM:',''), 
				 ('A) sample size:', self.panel.NT_before_loss), 
				 ('B) lost to GARCH/ARIMA:',self.panel.tot_lost_obs),
				 ('C) lost to FE/RE:', self.panel.number_of_RE_coef),
				 ('D) coefficients in regr:',self.panel.args.n_args), 
				 ('Degrees of freedom (A-B-C-D):',self.panel.df)
				 ]    
		
		tbl = [
			(samsum[0], None), 
			(samsum[1], samsum[4]), 
			(samsum[2], samsum[5]), 
			(samsum[3], samsum[6]),
			(None, None), 
			(grpremv[0], df[0]),
			(grpremv[1], df[1]),
			(grpremv[2], df[2]),
			(grpremv[3], df[3]),
			(grpremv[4], df[4]),
			(None, df[5]),
			(grpremv[5], None),
			(grpremv[6], None),
			(grpremv[7], None),
			(grpremv[8], None),
			(grpremv[9], None)
		
		]
		
		return heading+'\n' + self.parse_tbl(tbl)
		
	def define_table_size(self):
		"Defines the columns and width of the statistics tables"
		self.stat_left_col = max((16 + len(self.panel.input.Y_names[0]), 27))
		self.stat_right_col = 40
		self.stat_totlen = self.stat_left_col + self.stat_right_col + 2     
		

	def parse_tbl(self,tbl):  
		c =[self.stat_left_col, self.stat_right_col]  
		fmt = self.format_tbl_itm

		for i in range(len(tbl)):
			for j in [0,1]:
				itm = tbl[i][j]
				if itm is None:
					itm = ['', '']
				l = sum([len(str(itm[k])) for k in [0,1]]) + 2
				if l > c[j]: c[j] = l
					
		line = "="*(sum(c)+2) + '\n'  
		s = line
		for i in range(len(tbl)):
			for j in [0,1]:
				if tbl[i][j] is None:
					s += fmt('', '', c[j])
				else:
					s += fmt(tbl[i][j][0], tbl[i][j][1], c[j])
				if j==0:
					s += '  '
				else:
					s += '\n'
		s += line
		
		return s
 
		
	def format_tbl_itm(self, description, value, length):
		try:
			value = str(np.round(value, self.statistics_decimals))
		except:
			value = str(value)
		return "{:<{}}{}".format(description, length - len(value), value) 






class RegTableObj(dict):
	#This class handles the regression table itself
	def __init__(self, panel, comm, model_desc):
		dict.__init__(self)
		try:
			self.set(panel, comm, model_desc)
		except Exception as e:
			if not panel.options.supress_output:
				print(f'Exception while getting statistics: {e}')
		
	def set(self, panel, comm, model_desc):
		self.model_desc = model_desc
		self.Y_names = panel.input.Y_names
		self.X_names = panel.input.X_names
		self.args = comm.ll.args.dict_string
		self.n_variables = panel.args.n_args
		self.lags = panel.options.robustcov_lags_statistics[1]
		self.footer=f"\nSignificance codes: '=0.1, *=0.05, **=0.01, ***=0.001,    |=collinear\n\n{comm.ll.err_msg}"	
		self.dx_norm = comm.dx_norm
		self.t_stats(panel, comm.ll, comm.H, comm.G, comm.g, comm.constr)
		self.constraints_formatting(panel, comm.constr)    
		
		
		
	def t_stats(self, panel, ll, H, G, g, constr):
		self.d={'names':np.array(panel.args.caption_v),
		  				'names_int':panel.args.names_v,
						'count':range(self.n_variables),
						'args':ll.args.args_v}
		d = self.d
	
		T=len(d['names'])
		if H is None:
			return
		d['se_robust'],d['se_st']=sandwich(H, G, g, constr, panel, self.lags)
		d['se_robust_oposite'],d['se_st_oposite']=sandwich(H, G, g, constr, panel, self.lags,oposite=True)
		d['se_robust'][np.isnan(d['se_robust'])]=d['se_robust_oposite'][np.isnan(d['se_robust'])]
		d['se_st'][np.isnan(d['se_st'])]=d['se_st_oposite'][np.isnan(d['se_st'])]

		no_nan=np.isnan(d['se_robust'])==False
		valid=no_nan
		valid[no_nan]=(d['se_robust'][no_nan]>0)
		d['tstat']=np.array(T*[np.nan])
		d['tsign']=np.array(T*[np.nan])
		d['tstat'][valid]=d['args'][valid]/d['se_robust'][valid]
		d['tsign'][valid]=(1-stat_dist.tcdf(np.abs(d['tstat'][valid]),panel.df))#Two sided tests
		d['sign_codes']=get_sign_codes(d['tsign'])
		z = stat_dist.tinv025(panel.df)
		d['conf_low'] = d['args'] -z*d['se_robust']
		d['conf_high'] = d['args'] +z*d['se_robust']
		
	def constraints_formatting(self, panel, constr):
		mc_report={}
		if not constr is None:
			mc_report = constr.mc_report
		d=self.d
		if not self.dx_norm is None:
			d['dx_norm']=self.dx_norm
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

		for i in mc_report:#adding associates of non-severe multicollinearity
			d['multicoll'][i]='|'
			d['assco'][i]=panel.args.caption_v[mc_report[i]]	  

	def table(self,n_digits = 3,fmt = 'NORMAL',stacked = True, show_direction = False, show_constraints = True, show_confidence = False):
		include_cols,llength=self.get_cols(stacked, show_direction, show_constraints, show_confidence)
		if fmt=='INTERNAL':
			self.X=None
			return str(self.args),None
		self.include_cols=include_cols
		self.n_cols=len(include_cols)		
		for a, l,is_string,name,neg,just,sep,default_digits in TABLE_FORMAT:		
			self[a] = Column(self.d,a,l, is_string, name, neg, just, sep, default_digits,self.n_variables)
		self.X=self.output_matrix(n_digits)
		s = format_table(self.X, self.footer, self.d)
		return s,llength


	def output_matrix(self,digits):
		X=[['']*self.n_cols for i in range(self.n_variables+1)]
		for i in range(self.n_cols):
			a=self.include_cols[i]
			X[0][i]=self[a].name
			v=self[a].values(digits)
			for j in range(self.n_variables):
				X[j+1][i]=v[j]
		return X	


	def get_cols(self,stacked,
									show_direction,
									show_constraints, 
									show_confidence):
		"prints a single regression"
		dx_col=[]
		llength=9
		if show_direction:
			dx_col=['dx_norm']
		else:
			llength-=1
		mcoll_col=['multicoll']
		if show_constraints:
			mcoll_col=[ 'multicoll','assco','set_to', 'cause']
		conf_coll = []
		if show_confidence:
			conf_coll = ['conf_low','conf_high']
		else:
			llength-=2		
		if stacked:
			cols=['count','names', ['args','se_robust', 'sign_codes']] +conf_coll + dx_col + ['tstat', 'tsign'] + mcoll_col
		else:
			cols=['count','names', 'args','se_robust', 'sign_codes'] +conf_coll + dx_col + ['tstat', 'tsign'] + mcoll_col		
		return cols,llength


class Column:
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
		try:
			if self.length is None:
				if digits=='SCI':
					return self.input
				else:
					return np.round(self.input,digits)
			return np.round(self.input,self.length)
		except:
			if self.length is None:
				return self.input
			else:
				return np.array([str(i).ljust(self.length)[:self.length] for i in self.input])

def sandwich(H, G, g, constr, panel,lags,oposite=False,resize=True):
	H,G,idx=reduce_size(H, G, g, constr,oposite,resize)
	lags=lags+panel.lost_obs
	onlynans = np.array(len(idx)*[np.nan])
	try:
		hessin=np.linalg.inv(-H)
	except np.linalg.LinAlgError as e:
		print(e)
		return np.array(onlynans),np.array(onlynans)
	se_robust,se,V=stat.robust_se(panel,lags,hessin,G)
	se_robust,se,V=expand_x(se_robust, idx),expand_x(se, idx),expand_x(V, idx,True)
	if se_robust is None:
		se_robust = np.array(onlynans)
	if se is None:
		se = np.array(onlynans)
	return se_robust,se

def reduce_size(H, G, g, constr, oposite,resize):
	#this looks unneccessary complicated
	if constr is None:
		return H, G, np.ones(len(g),dtype=bool)
	if (G is None) or (H is None):
		return
	m=len(H)
	if not resize:
		return H,G,np.ones(m,dtype=bool)
	mc_report=constr.mc_report.keys()
	c = list(constr.fixed.keys())	
	if oposite:
		mc_report=[constr.mc_report[i] for i in constr.mc_report]
		c = []
		for i in constr.fixed:
			if not constr.fixed[i].assco_ix is None:
				c.append(constr.fixed[i].assco_ix)
	if False:#not sure why this is here
		for i in mc_report:
			if not i in c:
				c.append(i)
	idx=np.ones(m,dtype=bool)
	if len(c)>0:#removing fixed constraints from the matrix
		idx[c]=False
		H=H[idx][:,idx]
		G=G[:,:,idx]
	return H,G,idx

def expand_x(x,idx,matrix=False):
	x = np.real(x)
	m=len(idx)
	if matrix:
		x_full=np.zeros((m,m))
		x_full[:]=np.nan
		ref=np.arange(m)[idx]
		for i in range(len(x)):
			try:
				x_full[ref[i],idx]=x[i]
				x_full[idx,ref[i]]=x[i]
			except:
				a=0
	else:
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


class Statistics:
	def __init__(self, comm, panel, delta_time):
		self.diag = Diagnostics(comm, panel)
		self.info = Information(panel, comm, delta_time)

class Diagnostics:
	#This class handles the diagnostics of the regression	
	def __init__(self, comm, panel):
		ll = comm.ll
		ll.standardize(panel)

		self.Rsq_st, self.Rsqadj_st, self.LL_ratio,self.LL_ratio_OLS_st, self.F_st, self.F_p_st=stat.goodness_of_fit(ll,True,panel)	
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS, self.F, self.F_p=stat.goodness_of_fit(ll,False,panel)	
		self.no_ac_prob,self.rhos,self.RSqAC=stat.breusch_godfrey_test(panel,ll,10)
		self.DW, self.DW_no_panel=stat.DurbinWatson(panel,ll)
		self.JB_prob, self.JB, self.skewness, self.kurtosis, self.Omnibus, self.Omnibus_pval = stat.JB_normality_test(ll.u_long,panel)
		self.JB_prob_st, self.JB_st, self.skewness_st, self.kurtosis_st, self.Omnibus_st,  self.Omnibus_pval_st = stat.JB_normality_test(ll.e_RE_norm_centered_long,panel)
		self.ADF_stat,self.c1,self.c5=stat.adf_test(panel,ll,10)
		self.ci, self.n_ci = self.get_CI(comm.constr)

	def get_CI(self, constr):
		ci ='None'
		ci_n = 'None'
		if not constr is None:
			if not constr.ci is None:
				ci = np.round(constr.ci)
				ci_n = constr.ci_n
		return ci, ci_n
		
class Information:
	#This class handles the information about the regression
	def __init__(self, panel, comm, delta_time):

		model, method = self.get_model(panel)
		n,T,k = panel.X.shape
		rob = panel.options.robustcov_lags_statistics[1]>0
		run_time = np.round(delta_time)
		ll = comm.ll

		self.pqdkm=panel.pqdkm
		self.delta_time = delta_time
		self.instruments=panel.input.Z_names[1:]
		self.df=panel.df
		self.N,self.T,self.k = n,T,k
		self.dep_var       = panel.input.Y_names[0]
		self.model         = model
		self.method        = method
		self.date_str      = datetime.now().strftime('%a, %d %b %Y')
		self.time_str      = datetime.now().strftime('%H:%M:%S')
		self.run_time_str  = f'{run_time} ({comm.its}) [{comm.conv}]'
		self.obs_count     = panel.NT
		self.df_model      = k - 1
		self.cov_type      = rob
		self.log_lik       = ll.LL
		self.aic           = 2 * k - 2 * comm.ll.LL
		self.bic           = panel.NT * k - 2 * comm.ll.LL
		self.panel_groups  = n
		self.panel_dates   = T
		self.initvar 	   = ll.args.args_v[panel.args.names_v.index(INITVAR)]

	def get_model(self, panel):
		p, q, d, k, m = panel.pqdkm
		s = ""
		if d>0:
			s+=f"ARIMA{(p, d, q)}"
		elif (p*q):
			s+=f"ARMA{(p,q)}"
		elif (p==0)&(q>0):
			s+=f"MA{q}"
		elif (p>0)&(q==0):
			s+=f"AR{p}"

		if (k*m):
			if s!='': s += ', '
			s += f"GARCH{(k,m)}"
		t = panel.options.fixed_random_time_eff
		i = panel.options.fixed_random_group_eff
		if t+i:
			if s!='': s += ' '      
			if (t==2)&(i==2):
				s += '2-way RE'
			elif (t==1)&(i==1):
				s += '2-way FE'  
			elif (t==2)|(i==2):
				s += 'FE'  
			elif (t==1)|(i==1):
				s += 'RE'     
				
		if s == '':
			s = 'OLS'
			method = 'Least Squares'
		else:
			method = 'Maximum Likelihood'
		
		return s, method




def format_table(X, tail, d):

	remove_init_var(X, d)
	
	p=''
	X = np.array(X)
	dcol = 2
	n,k = X.shape
	colwith = [max(
					[len(s) for s in X[:,i]])
									 for i in range(k)
													 ]
	
	

	for i in range(n):
		p+='\n'
		for j in range(k):
			if j == 0:
				p+=f'{X[i][j]}'.ljust(3)
			elif j==1:
				p+=f'{X[i][j]}'.ljust(colwith[j]+dcol)
			else:
				p+=f'{X[i][j]}'.rjust(colwith[j]+dcol)
	p = p.split('\n')
	n = len(p[1])
	p[0] = '='*n
	p = p[:2] + ['-'*n] + p[2:] + ['='*n]
	p = '\n'.join(p)
	p = 'Regression results:\n' + p
	return p + tail

		

def remove_init_var(X,d):
	if not INITVAR in d['names_int']:
		return
	i = d['names_int'].index(INITVAR)+1
	X.pop(i)



TABLE_FORMAT =[
#python variable name,	length,		is string,  display name,		neg. values,	justification	next tab space		round digits (None=no rounding,-1=set by user)
['count',				2,			False,		'',					False,			'right', 		2,					None],
['names',				None,		True,		'Variable names',	False,			'right', 		2, 					None],
['args',				None,		False,		'coef',				True,			'right', 		2, 					-1],
['se_robust',			None,		False,		'std err',			True,			'right', 		3, 					-1],
['sign_codes',			5,			True,		'',					False,			'left', 		2, 					-1],
['dx_norm',				None,		False,		'direction',		True,			'right', 		2, 					None],
['tstat',				2,			False,		't',				True,			'right', 		2, 					2],
['tsign',				None,		False,		'P>|t|',			False,			'right', 		2, 					3],
['conf_low',			None,		False,		'[0.025',			False,			'right', 		2, 					3],
['conf_high',			None,		False,		'0.975]',			False,			'right', 		2, 					3],
['multicoll',			1,			True,		'',					False,			'left', 		2, 					None],
['assco',				20,			True,		'collinear with',	False,			'center', 		2, 					None],
['set_to',				6,			True,		'set to',			False,			'center', 		2, 					None],
['cause',				50,			True,		'cause',			False,			'right', 		2, 					None]]		