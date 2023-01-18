#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module calculates statistics and saves it to a file

import numpy as np
import stat_dist
import stat_functions as stat
import time

STANDARD_LENGTH=8
		


class output:
	def __init__(self,ll,panel,comm, dx_norm):
		self.ll=ll
		self.comm=comm
		self.panel=panel
		self.lags=self.panel.options.robustcov_lags_statistics.value[1]
		self.n_variables=self.panel.args.n_args
		self.incr=0
		self.dx_norm = dx_norm
		self.d={'names':np.array(self.panel.args.names_v),
				'count':range(self.n_variables),
				'args':self.ll.args.args_v}
		self.update(comm, 0, ll, self.incr, dx_norm)
		self.heading()

	def update(self,comm, its, ll,incr, dx_norm):
		self.comm=comm
		self.iterations=its
		self.ll=ll
		self.dx_norm = dx_norm
		self.incr=incr
		self.d['args']=ll.args.args_v
		self.constraints_printout()
		self.t_stats()
		self.heading()
		
	def statistics(self):
		return Statistics(self.ll,self.panel)

	def reg_table(self):
		return RegTableObj(self)
	
	def t_stats(self):
		d=self.d
		comm=self.comm
		panel=self.panel
		
		T=len(d['names'])
		if comm.H is None:
			return
		d['se_robust'],d['se_st']=sandwich(comm,self.lags)
		d['se_robust_oposite'],d['se_st_oposite']=sandwich(comm,self.lags,oposite=True)
		if not (d['se_st_oposite'] is None):
			d['se_robust'][np.isnan(d['se_robust'])]=d['se_robust_oposite'][np.isnan(d['se_robust'])]
			d['se_st'][np.isnan(d['se_st'])]=d['se_st_oposite'][np.isnan(d['se_st'])]
		#d['se_robust_fullsize'],d['se_st_fullsize']=sandwich(comm,self.lags,resize=False)
		no_nan=np.isnan(d['se_robust'])==False
		valid=no_nan
		valid[no_nan]=(d['se_robust'][no_nan]>0)
		d['tstat']=np.array(T*[np.nan])
		d['tsign']=np.array(T*[np.nan])
		d['tstat'][valid]=d['args'][valid]/d['se_robust'][valid]
		d['tsign'][valid]=(1-stat_dist.tcdf(np.abs(d['tstat'][valid]),panel.df))#Two sided tests
		d['sign_codes']=get_sign_codes(d['tsign'])
	
	def heading(self):
		if self.comm.constr.CI is None:
			CI ='None'
		else:
			CI = np.round(self.comm.constr.CI)
		n_CI=len(self.comm.constr.mc_problems)
		
		
		self.model_desc =  f"Model:\t{self.panel.input.Y_names[0]} = {' + '.join(self.panel.input.X_names)}\n"
		instr='None'
		if not self.panel.input.Z_names is None:
			self.model_desc += f"Instruments:\t{', '.join(self.panel.input.Z_names[1:])}\n" 
			
		n,T,k=self.panel.X.shape
		run_time = np.round(time.time()-self.comm.start_time)

		s = (
		f"LL:\t{self.ll.LL        }\tIteration:\t{self.iterations        }\tPanel observations:\t{self.panel.NT_before_loss }\n"
		f"Increment:\t{self.incr  }\tCollinearity - max(cond.idx):\t{CI  }\tPanel dates:\t{T                                }\n"
		f"Time elapsed:\t{run_time}\tCollinearity - # variables:\t{n_CI  }\tPanel groups:\t{n                               }\n"
	
		
		)
		
		self.output_stats=s

		
	def constraints_printout(self):
		panel=self.panel
		comm=self.comm
		constr=comm.constr
		weak_mc_dict=comm.constr.weak_mc_dict
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
				
		for i in weak_mc_dict:#adding associates of non-severe multicollinearity
			d['multicoll'][i]='|'
			d['assco'][i]=panel.args.names_v[weak_mc_dict[i][0]]	

	
class RegTableObj(dict):
	def __init__(self,output):
		dict.__init__(self)
		self.d=output.d
		self.Y_names=output.panel.input.Y_names
		self.X_names=output.panel.input.X_names
		self.args=output.ll.args.dict_string
		self.n_variables=output.n_variables
		self.output_stats = output.output_stats
		self.model_desc = output.model_desc
		self.footer=f"\n\nSignificance codes: '=0.1, *=0.05, **=0.01, ***=0.001,    |=collinear\n\n{output.ll.err_msg}"	
	
	def table(self,n_digits = 3,brackets = '(',fmt = 'NORMAL',stacked = True, show_direction = False, show_constraints = True):
		include_cols,llength=self.get_cols(stacked, show_direction, show_constraints)
		if fmt=='INTERNAL':
			self.X=None
			return str(self.args),None
		self.include_cols=include_cols
		self.n_cols=len(include_cols)		
		for a, l,is_string,name,neg,just,sep,default_digits in pr:		
			self[a]=column(self.d,a,l, is_string, name, neg, just, sep, default_digits,self.n_variables)
		self.X=self.output_matrix(n_digits,brackets)
		s=format_table(self.X, include_cols,fmt,
					   "Paneltime GARCH-ARIMA panel regression",
					   self.model_desc, 
					   self.output_stats,
					   self.footer)
		return s,llength

	
	def output_matrix(self,digits,brackets):
		structured=False
		for i in range(self.n_cols):
			if type(self.include_cols[i])==list:
				structured=True
				break
		if structured:
			return self.output_matrix_structured(digits, brackets)
		else:
			return self.output_matrix_flat(digits, brackets)
		
		
	def output_matrix_structured(self,digits,brackets):
		X=[['']*self.n_cols for i in range(3*(self.n_variables+1)-1)]
		for i in range(self.n_cols):
			a=self.include_cols[i]
			if type(a)==list:
				h=self[a[0]].name.replace(':',' ')
				if brackets=='[':
					X[0][i]=f"{h}[{self[a[1]].name}]:"
				elif brackets=='(':
					X[0][i]=f"{h}({self[a[1]].name}):"
				else:
					X[0][i]=f"{h}/{self[a[1]].name}:"
				v=[self[a[j]].values(digits) for j in range(3)]
				for j in range(self.n_variables):
					X[(j+1)*3-1][i]=v[0][j]
					if brackets=='[':
						X[(j+1)*3][i]=f"[{v[1][j]}]{v[2][j]}"
					elif brackets=='(':
						X[(j+1)*3][i]=f"({v[1][j]}){v[2][j]}"
					else:
						X[(j+1)*3][i]=f"{v[1][j]}{v[2][j]}"
			else:
				X[0][i]=self[a].name
				v=self[a].values(digits)
				for j in range(self.n_variables):
					X[(j+1)*3-1][i]=v[j]
		return X	
	
	def output_matrix_flat(self,digits,brackets):
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
				show_constraints):
		"prints a single regression"
		dx_col=[]
		llength=9
		if show_direction:
			dx_col=['dx_norm']
		else:
			llength-=1
		mcoll_col=[]
		if show_constraints:
			mcoll_col=[ 'multicoll','assco','set_to', 'cause']
		else:
			llength-=2		
		if stacked:
			cols=['count','names', ['args','se_robust', 'sign_codes']] + dx_col + ['tstat', 'tsign'] + mcoll_col
		else:
			cols=['count','names', 'args','se_robust', 'sign_codes'] + dx_col + ['tstat', 'tsign'] + mcoll_col		
		return cols,llength
	
	
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

def sandwich(comm,lags,oposite=False,resize=True):
	panel=comm.panel
	H,G,idx=reduce_size(comm,oposite,resize)
	lags=lags+panel.lost_obs
	try:
		hessin=np.linalg.inv(-H)
	except Exception as e:
		print(e)
		return None,None
	se_robust,se,V=stat.robust_se(panel,lags,hessin,G)
	se_robust,se,V=expand_x(se_robust, idx),expand_x(se, idx),expand_x(V, idx,True)
	return se_robust,se

def reduce_size(comm,oposite,resize):
	if comm.constr is None:
		return comm.H, comm.G, np.ones(len(comm.g),dtype=bool)
	H=comm.H
	G=comm.G
	if (G is None) or (H is None):
		return
	m=len(H)
	if not resize:
		return H,G,np.ones(m,dtype=bool)
	weak_mc_dict=comm.constr.weak_mc_dict.keys()
	constr=list(comm.constr.fixed.keys())	
	if oposite:
		weak_mc_dict=[comm.constr.weak_mc_dict[i][0] for i in comm.constr.weak_mc_dict]
		constr=[]
		for i in comm.constr.fixed:
			if not comm.constr.fixed[i].assco_ix is None:
				constr.append(comm.constr.fixed[i].assco_ix)
	for i in weak_mc_dict:
		if not i in constr:
			constr.append(i)
	idx=np.ones(m,dtype=bool)
	if len(constr)>0:#removing fixed constraints from the matrix
		idx[constr]=False
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
		
	
class Statistics:
	def __init__(self,ll,panel):
		ll.standardize(panel)
		self.df=panel.df
		self.N,self.T,self.k=panel.X.shape
		self.Rsq_st, self.Rsqadj_st, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(ll,True,panel)	
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(ll,False,panel)	
		self.no_ac_prob,self.rhos,self.RSqAC=stat.breusch_godfrey_test(panel,ll,10)
		self.DW=stat.DurbinWatson(panel,ll)
		self.norm_prob=stat.JB_normality_test(ll.e_norm,panel)
		self.ADF_stat,self.c1,self.c5=stat.adf_test(panel,ll,10)
		self.df_str=self.gen_df_str(panel)	
		self.instruments=panel.input.Z_names[1:]
		self.pqdkm=panel.pqdkm
		
	def gen_df_str(self,panel):
		summary=f"""
  SAMPLE SIZE SUMMARY:
\tOriginal sample size\t\t:\t{panel.orig_size}
\tSample size after filtering\t\t:\t{panel.NT_before_loss}
\tDegrees of freedom\t\t:\t{panel.df}
\tNumber of IDs\t\t:\t{self.N:,}
\tNumber of dates (maximum)\t\t:\t{self.T}\n"""		
			
		group_rmv=f"""
  REMOVED GROUPS BECAUSE OF TOO FEW OBSERVATIONS:
\tObservations per group lost because of
\tA)\tARIMA/GARCH\t:\t{panel.lost_obs}
\tB)\tMin # of obs in user preferences:\t:\t{panel.options.min_group_df.value}
\tMin # observations required (A+B)\t\t:\t{panel.lost_obs+panel.options.min_group_df.value}\n
\tGroups removed
\tA)\tTotal # of groups\t:\t{len(panel.idincl)}
\tB)\t# of groups removed\t:\t{sum(panel.idincl==False)}
\t# of groups remaining (A-B)\t\t:\t{sum(panel.idincl==True)}

\t# of observations removed\t\t:\t{panel.input.X.shape[0]-panel.NT_before_loss}\n"""
		
		
		s=f"""
{summary}
{group_rmv}
  DEGREES OF FREEDOM:
\tA)\tSample size\t:\t{panel.NT_before_loss}
\tB)\tObservations lost to
\t\tGARCH/ARIMA\t:\t{panel.tot_lost_obs}	
\tRandom/Fixed Effects in
\tC)\tMean process\t:\t{panel.number_of_RE_coef}
\tD)\tVariance process\t:\t{panel.number_of_RE_coef_in_variance}
\tE)\tNumber of coefficients in
\t\tRegression\t:\t{panel.args.n_args:,}
\tDegrees of freedom (A-B-C-D-E)\t\t:\t{panel.df}\n\n"""
		return s
	
	def gen_mod_fit(self,n_digits = 3):
		return f"""	
\tLL-ratio\t\t:\t{round(self.LL_ratio,n_digits)}
\tR-squared (from observed data)\t\t:\t{round(self.Rsq*100,2)}%
\tAdjusted R-squared (from observed data)\t\t:\t{round(self.Rsqadj*100,2)}%
\tR-squared (from normalized data)\t\t:\t{round(self.Rsq_st*100,2)}%
\tAdjusted R-squared (from normalized data)\t\t:\t{round(self.Rsqadj_st*100,2)}%

\t("Normalized data" means that the data is adjusted with the estimated ARIMA-GARCH parameters and random/fixed effects.)

\tDurbin-Watson statistic:\t\t:\t{round(self.DW,2)}
\tBreusch-Godfrey test\t\t:\t{round(self.no_ac_prob*100,n_digits)}% \t(significance, probability of no auto correlation)
\tJarque–Bera test for normality\t\t:\t{round(self.norm_prob*100,n_digits)}% \t(significance, probability of normality)\n
"""		


	def adf_str(self,n_digits = 3):
			
		if not self.ADF_stat=='NA':
			if self.ADF_stat<self.c1:
				self.ADF_res="Unit root rejected at 1%"
			elif self.ADF_stat<self.c5:
				self.ADF_res="Unit root rejected at 5%"
			else:
				self.ADF_res="Unit root not rejected"		
			adf=f"""
\tAugmented Dicky-Fuller (ADF)        
\t\tTest statistic\t:\t{round(self.ADF_stat,n_digits)}
\t\t1% critical value\t:\t{round(self.c1,n_digits)}
\t\t5% critical value\t:\t{round(self.c5,n_digits)}
\t\tResult\t:\t\t{self.ADF_res}
			"""
		else:
			adf="Unable to calculate ADF"
		if self.df<1:
			s+="""
\tWARNING: All your degrees of freedom (df) has been consumed, so statistics cannot be computed.
\tyou can increase df by for example turning off random/fixed effects """
		return adf
	
	
def get_tab_stops(X,f):
	
	m_len = f.measure("m")
	counter=2*m_len
	tabs=[f"{counter}",'numeric']
	r,c=np.array(X).shape
	for i in range(c):
		t=1
		num_max=0
		for j in range(r):
			s=str(X[j][i])
			if '.' in s:
				a=s.split('.')
				num_max=max((len(a[0]),num_max))
			t=max((f.measure(X[j][i])+(num_max+2)*m_len,t))
		counter+=t
		tabs.extend([f"{counter}",'numeric'])			
	return tabs

l=STANDARD_LENGTH
#python variable name,	length,		is string,  display name,		neg. values,	justification	next tab space		round digits (None=no rounding,-1=set by user)
pr=[
		['count',		2,			False,		'',					False,			'right', 		2,					None],
		['names',		None,		True,		'Variable names:',	False,			'right', 		2, 					None],
		['args',		None,		False,		'Coef:',			True,			'right', 		2, 					-1],
		['se_robust',	None,		False,		'rob.SE',			True,			'right', 		3, 					-1],
		['sign_codes',	5,			True,		'',					False,			'left', 		2, 					-1],
		['dx_norm',		None,		False,		'direction:',		True,			'right', 		2, 					None],
		['tstat',		2,			False,		't-stat.:',			True,			'right', 		2, 					2],
		['tsign',		None,		False,		'p-value:',			False,			'right', 		2, 					3],
		['multicoll',	1,			True,		'',					False,			'left', 		2, 					None],
		['assco',		20,			True,		'collinear with',	False,			'center', 		2, 					None],
		['set_to',		6,			True,		'set to',			False,			'center', 		2, 					None],
		['cause',		50,			True,		'cause',			False,			'right', 		2, 					None]]		

		
def format_table(X,cols,fmt,heading, mod, stats,tail):
	if fmt=='NORMAL':
		return mod + stats+format_normal(X,[1],cols)+tail
	if fmt=='LATEX':
		return mod + stats+format_latex(X,cols,heading)+tail
	if fmt=='HTML':
		return format_html(X,cols,heading,stats, mod)+tail	
	
	
def format_normal(X,add_rows=[],cols=[]):
	p=''
	if 'multicoll' in cols:
		constr_pos=cols.index('multicoll')+1
		p="\t"*constr_pos+"constraints:".center(38)
	p+="\n"
	for i in range(len(X)):
		p+='\n'*(i in add_rows)
		p+='\n'
		for j in range(len(X[0])):
			p+=f'\t{X[i][j]}'
		
	return p	


def format_latex(X,cols,heading):
	X=np.array(X,dtype='U128')
	n,k=X.shape
	p="""
\\begin{table}[ht]
\\caption{%s} 
\\centering
\\begin{tabular}{""" %(heading,)
	p+=' c'*k+' }\n\\hline'
	p+='\t'+' &\t'.join(X[0])+'\\\\\n\\hline\\hline'
	for i in range(1,len(X)):
		p+='\t'+ ' &\t'.join(X[i])+'\\\\\n'
	p+="""
\\hline %inserts single line
\\end{tabular}
\\label{table:nonlin} % is used to refer this table in the text
\\end{table}"""
	return p	

def format_html(X,cols,heading,head, mod):
	X=np.array(X,dtype='U128')
	n,k=X.shape
	if head[-1:] == '\n':
		head = head[:-1]
	head = head.replace('\n',"</td></tr>\n<td class='h'>")
	head = head.replace('\t',"</td><td class='h'>")
	mod = mod.replace('\n','<br>')
	spaces = '&nbsp'*4
	p=(f"<br><br><h1>{heading}</h1><br><br>"
	f"<p>{mod}<br>"
	f"<table class='head'></tr><td class='h'>{head}</td></table><br><br>\n\n"
	f"<p><table>\t<tr><th>"
	)
	p += ('\t</th><th>'+spaces).join(X[0])+'</th></tr>\n'
	for i in range(1,len(X)):
		p+='\t<tr><td>'+'\t</td><td>'.join(X[i])+'</td></tr>\n'
	p+='</table></p>'
	return p		
	
alphabet='abcdefghijklmnopqrstuvwxyz'
class join_table(dict):
	"""Creates a  joint table of several regressions with columns of the join_table_column class.
	See join_table_column for data handling."""
	def __init__(self,args,panel,varnames=[]):
		dict.__init__(self)
		self.names_category_list=list([list(i) for i in args.names_category_list])#making a copy
		k=0
		for i in varnames:
			if i in self.names_category_list[0]:
				k=self.names_category_list[0].index(i)+1
			else:
				self.names_category_list[0].insert(k,i)
				k+=1
		self.names_v=[itm for s in self.names_category_list for itm in s]#flattening
		
	def update(self,ll,stats,desc,panel):
		if not desc in self:
			for i in range(len(ll.args.names_category_list)):
				for j in ll.args.names_category_list[i]:
					if not j in self.names_category_list[i]:
						self.names_category_list[i].append(j)
			self.names_v=[itm for s in self.names_category_list for itm in s]#flattening
		self[desc]=join_table_column(stats, ll,panel)
		
		
	def make_table(self, stacked, brackets,digits,caption):
		keys=list(self.keys())
		k=len(keys)
		n=len(self.names_v)
		if stacked:
			X=[['' for j in range(2+k)] for i in range(4+3*n)]
			for i in range(n):
				X[3*i+1][1]=self.names_v[i]		
				X[3*i+1][0]=i
			X[1+3*n][1]='Log likelihood'
			X[2+3*n][1]='Degrees of freedom'	
			X[3+3*n][1]='Adjusted R-squared'	
		else:
			X=[['' for j in range(2+2*k)] for i in range(4+n)]
			for i in range(n):
				X[i+1][1]=self.names_v[i]	
				X[i+1][0]=i
			X[1+n][1]='Log likelihood'
			X[2+n][1]='Degrees of freedom'		
			X[3+n][1]='Adjusted R-squared'	
		for i in range(k):
			self.make_column(i,keys[i],X,stacked, brackets,digits,caption)
		s=format_normal(X,[1,(1+stacked*2)*n+1,(1+stacked*2)*n+4])
		s+=f"\n\nSignificance codes: '=0.1, *=0.05, **=0.01, ***=0.001,    |=collinear\n"
		max_mod=0
		models=[]
		for i in range(len(keys)):
			key=self[keys[i]]
			p,q,d,k,m=key.pqdkm
			models.append(f"\n{alphabet[i]}: {keys[i]}")
			max_mod=max(len(models[i]),max_mod)
		for i in range(len(keys)):
			s+=models[i].ljust(max_mod+2)
			if len(key.instruments):
				s+=f"\tInstruments: {key.instruments}"
			s+=f"\tARIMA({p},{d},{q})-GARCH({k},{m})"
		return s,X
			
	def make_column(self,col,key,X,stacked, brackets,digits,caption):
		if not 'se_robust' in self[key].stats:
			return
		
		if caption=='JOINED LONG':
			X[0][(2-stacked)*col+2]+=f"{self[key].Y_name} ({alphabet[col]})"
		else:
			X[0][(2-stacked)*col+2]=alphabet[col]
		n=len(self.names_v)
		m=len(self[key].args.names_v)
		ix=[self.names_v.index(i) for i in self[key].args.names_v]
		se=np.round(self[key].stats['se_robust'],digits)
		sgn=self[key].stats['sign_codes']
		args=np.round(self[key].args.args_v,digits)
		if brackets=='[':
			se_sgn=[f"[{se[i]}]{sgn[i]}" for i in range(m)]
		elif brackets=='(':
			se_sgn=[f"({se[i]}){sgn[i]}" for i in range(m)]
		else:
			se_sgn=[f"{se[i]}{sgn[i]}" for i in range(m)]				
		if stacked:
			for i in range(m):
				X[3*ix[i]+1][col+2]=args[i]
				X[3*ix[i]+2][col+2]=se_sgn[i]
			X[1+3*n][col+2]=round_sign_digits(self[key].LL,5,1)
			X[2+3*n][col+2]=self[key].df	
			X[3+3*n][col+2]=f"{round(self[key].Rsqadj*100,1)}%"
		else:
			for i in range(m):
				X[ix[i]+1][col*2+2]=args[i]
				X[ix[i]+1][col*2+3]=se_sgn[i]		
			X[1+n][col*2+3]=round_sign_digits(self[key].LL,5,1)
			X[2+n][col*2+3]=self[key].df
			X[3+n][col*2+3]=f"{round(self[key].Rsqadj*100,1)}%"
	


class join_table_column:
	def __init__(self,stats,ll,panel):
		self.stats=stats
		self.LL=ll.LL
		self.df=panel.df
		self.args=ll.args
		self.Rsq, self.Rsqadj, self.LL_ratio,self.LL_ratio_OLS=stat.goodness_of_fit(ll,True,panel)
		self.instruments=panel.input.Z_names[1:]
		self.pqdkm=panel.pqdkm		
		self.Y_name=panel.input.Y_names
		
		
		
		
		
def round_sign_digits(x,digits,min_digits=0):
	d=int(np.log10(abs(x)))
	return np.round(x,max((digits-1,d+min_digits))-d)
