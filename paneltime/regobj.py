#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains classes used in the regression





import statproc as stat
import numpy as np
import time
import threading
import debug
import regprocs as rp
import functions as fu
import calculus


min_AC=0.000001


def posdef(a,da):
	return list(range(a,a+da)),a+da

class panel:
	def __init__(self,p,d,q,m,k,X,Y,groups,x_names,y_name,groups_name,fixed_random_eff,args,W,w_names,master,descr,data,h,has_intercept):
		"""
		No effects    : fixed_random_eff=0\n
		Fixed effects : fixed_random_eff=1\n
		Random effects: fixed_random_eff=2\n
		
		"""
		if groups_name is None:
			fixed_random_eff=0

		self.initial_defs(h,X,groups,W,has_intercept,data,p,q,m,k,d,x_names,y_name,groups_name,w_names,master,descr,fixed_random_eff)
		
		self.X,self.Y,self.W,self.max_T,self.T_arr,self.N=self.arrayize(X, Y, W, groups)

		self.masking()
		self.lag_variables(max((q,p,k,m)))
		
		self.name_vector=self.make_namevector()
		self.gradient=calculus.gradient(self)
		self.hessian=calculus.hessian(self,master)		
		
		self.final_defs(p,d,q,m,k,X)
		
		self.between_group_ols()
		self.args=arguments(p, d, q, m, k, self, args,self.has_intercept)
		self.LL_restricted=LL(self.args.args_restricted, self).LL
		self.LL_OLS=LL(self.args.args_OLS, self).LL		

		
		


	def masking(self):
		self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
		#"initial observations" mask: 
		self.a=np.array([self.date_counter<self.T_arr[i] for i in range(self.N)])# sets observations that shall be zero to zero by multiplying it with the arrayized variable
	
		#"after lost observations" masks: 
		self.included=np.array([(self.date_counter>=self.lost_obs)*(self.date_counter<self.T_arr[i]) for i in range(self.N)])# sets observations that shall be zero after lost observations to zero by multiplying it with the arrayized variable
		self.n_i=np.sum(self.included,1).reshape((self.N,1,1))#number of observations for each i
		self.n_i=self.n_i+(self.n_i<=0)#ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case	
		
	def initial_defs(self,h,X,groups,W,has_intercept,data,p,q,m,k,d,x_names,y_name,groups_name,w_names,master,descr,fixed_random_eff):
		rp.redefine_h_func(h)
		self.has_intercept=has_intercept
		self.data=data
		self.lost_obs=np.max((p,q))+max((m,k))+d+3
		self.x_names=x_names
		self.y_name=y_name
		self.groups_name=groups_name
		self.w_names=w_names		
		self.p,self.d,self.q,self.m,self.k,self.nW,self.n_beta=p,d,q,m,k,W.shape[1],X.shape[1]
		self.n_beta=len(X[0])
		self.descr=descr
		self.its_reg=0
		self.FE_RE=fixed_random_eff
		self.groups=groups		
		
	def final_defs(self,p,d,q,m,k,X):
		self.W_a=self.W*self.a
		self.tot_lost_obs=self.lost_obs*self.N
		self.NT_afterloss=np.sum(self.included)
		self.NT=self.NT_afterloss+self.tot_lost_obs				
		self.len_args=p+d+q+m+k+self.W.shape[2]+self.X.shape[2]+2*(m==0)
		self.number_of_RE_coef=self.N
		self.number_of_FE_coef_in_variance=self.N
		self.df=self.NT_afterloss-self.len_args-self.number_of_RE_coef-self.number_of_FE_coef_in_variance
		self.xmin=np.min(X,0).reshape((1,X.shape[1]))
		self.xmax=np.max(X,0).reshape((1,X.shape[1]))

	

	def make_namevector(self):
		"""Creates a vector of the names of all regression varaibles, including variables, ARIMA and GARCH terms"""
		p,q,m,k,nW=self.p,self.q,self.m,self.k,self.nW
		names=self.x_names[:]#copy variable names
		rp.add_names(p,'AR term ',names)
		rp.add_names(q,'MA term ',names)
		rp.add_names(m,'MACH term ',names)
		rp.add_names(k,'ARCH term ',names)
		names.extend(self.w_names)
		names.extend(['Variance (group eff.)'])
		names.extend(['z in h(e,z)'])
		return names
	
	def between_group_ols(self):
		N,T,k=self.X.shape
		if self.FE_RE==0:
			return
		if k+4>N and self.FE_RE==2:
			self.FE_RE=1
			print ("""Warning: Only %s groups compared to %s variables. Group mean regression 
				            does not have sufficient degrees of freedom for a reasonably robust variance
				            estimate. A fixed effect model will be run instead.""" %(N,k))	
			return
		g_X=np.sum(self.X*self.included,1)/self.n_i.reshape(N,1)
		g_Y=np.sum(self.Y*self.included,1)/self.n_i.reshape(N,1)
		non_zeros=np.sum(g_X**2,0)>0   #sometimes differencing may cause the group variables to be all zero.
		g_X=g_X[:,non_zeros]           #These variables need to be eliminated.
		beta,self.grp_err=stat.OLS_simple(g_Y,g_X)
		self.grp_v=np.var(self.grp_err)

	def lag_variables(self,max_lags):
		self.L=rp.make_lag_matrices(self.max_T,max_lags)
		self.I=np.diag(np.ones(self.max_T))
		self.zero=self.I*0
		if self.d==0:
			return
		Ld=(self.I-self.L[0])
		for i in range(1,self.d):
			Ld=fu.dot(self.I-L[0],self.Ld)		
		self.Y=fu.dot(Ld,self.Y)*self.a	
		self.X=fu.dot(Ld,self.X)*self.a
		if self.has_intercept:
			self.X[:,:,0]=1
		self.Y[:,:self.d]=0
		self.X[:,:self.d]=0		

	def LL(self,args):
		"""runs the LL"""
		self.its_reg+=1
		ll=LL(args,self)
		if not ll.LL is None:
			return ll

	def get_direction(self,ll,mc_limit,add_one_constr,dx_conv,has_problems,k,its):

		g,G=self.gradient.get(ll,return_G=True)
		hessian=self.hessian.get(ll)

		dc,constrained,reset,out,constr=self.solve(add_one_constr, G, g, hessian, ll, mc_limit, 
				                     dx_conv, has_problems,k,its)
		
		#fixing positive definit hessian (convexity problem) by using robust sandwich estimator
		if np.sum(dc*(constrained==0)*g)<0:
			#print("Warning: negative slope. Using robust sandwich hessian matrix to ensure positivity")
			hessin=rp.sandwich(hessian,G,0)
			for i in range(len(hessin)):
				hessin[i,i]=hessin[i,i]+(hessin[i,i]==0)
			hessian=-np.linalg.inv(hessin)
			dc,constrained,reset,out,constr=self.solve(add_one_constr, G, g, hessian, ll, mc_limit, 
						              dx_conv, has_problems,k,its)			
			
		if len(constr.constraints)>0:
			out.print()		
		return dc,g,G,hessian,constrained,reset
	
	def solve(self,add_one_constr,G,g,hessian,ll,mc_limit,dx_conv,has_problems,k,its):
		if not hasattr(self,'constr'):
			self.constr=None		
		constr=constraints(self.args,self.constr,add_one_constr)
		self.constr=constr
		if len(self.args.positions['z'])>0:
			constr.add(self.args.positions['z'][0],1e-15,10000)	
		hessian,reset,out=rp.handle_multicoll(G,self,ll.args_v,self.name_vector,constr,mc_limit,dx_conv,hessian,has_problems,k,its)
		dc,constrained=rp.solve(constr,hessian, g, ll.args_v)
		#constr_added=self.cond_testing(ll.args_v+dc,constr, 'lambda')
		#constr_added=constr_added or self.cond_testing(ll.args_v+dc,constr, 'gamma')
		#if constr_added:
		#	dc,constrained=rp.solve(constr,hessian, g, ll.args_v)
		return dc,constrained,reset,out,constr
	
	def cond_testing(self,args,constr,varname):
		c_added=False
		v=args[self.args.positions[varname]]
		k=len(v)
		m=9+k
		rows,cols=np.indices((m,m))
		if varname=='gamma':
			s=-1
		else:
			s=1
		X=self.I[0:m,0:m]
		if np.all(v<20) and np.all(v>-20):
			for i in range(k):
				X=X+np.diag(np.ones(m-i-1),-i-1)*v[i]*s
			c=np.linalg.cond(X)
			if c<=m:
				return False
		for i in range(1,k):
			constr.add(self.args.positions[varname][i],0)
			c_added=True
		row_vals=np.diag(rows,-1)
		col_vals=np.diag(rows,-1)
		d=v[0]
		z=np.sign(d)
		X=self.I[0:m,0:m]
		if k>1:
			X[row_vals,col_vals]=s*d
			c=np.linalg.cond(X)
			if c<=m: 
				return c_added
		if abs(d)>1.4:
			d=z*1.4
			X[row_vals,col_vals]=s*d
			c=np.linalg.cond(X)
			if c<=m: 
				constr.add(self.args.positions[varname][0],0,d)
				return True		
		while 1:
			d=d-z*0.05
			if z*d<0:
				raise RuntimeError('This should not happen')
			X[row_vals,col_vals]=s*d
			c=np.linalg.cond(X)
			if c<m:
				constr.add(self.args.positions[varname][0],0,d)
				return True
		raise RuntimeError('This should not happen')

	def params_ok(self,args):
		a=self.q_sel,self.p_sel,self.M_sel,self.K_sel
		for i in a:
			if len(i)>0:
				if np.any(np.abs(args[i])>0.999):
					return False
		return True

	def set_garch_arch(self,args,LL):
		p,q,m,k,nW=self.p,self.q,self.m,self.k,self.nW
		X=self.I+self.lag_matr(q,args['lambda'])
		try:
			AMA_1=np.linalg.inv(X)
		except:
			return None
		AAR=self.I-self.lag_matr(p,args['rho'])
		AMA_1AR=fu.dot(AMA_1,AAR)
		X=self.I-self.lag_matr(k,args['gamma'])
		try:
			GAR_1=np.linalg.inv(X)
		except:
			return None
		GMA=self.lag_matr(m,args['psi'])	
		GAR_1MA=fu.dot(GAR_1,GMA)
		return AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA

	def lag_matr(self,k,args):
		L=self.L
		if k==0:
			return self.zero
		a=[]
		for i in range(k):
			a.append(args[i]*L[i])
		a=np.sum(a,0)
		return a

	def de_arrayize(self,X,init_obs):
		"""X is N x T x k"""
		(N,T,k)=X.shape
		x_arr=[X[i,init_obs:self.T_arr[i][0]] for i in range(N)]
		ret=np.concatenate(x_arr,0)
		return ret	

	def arrayize(self,X,Y,W,groups):
		"""Splits X and Y into an arry of equally sized matrixes rows equal to the largest for each groups,
		and returns the matrix arrays and their row number"""
		NT,k=X.shape
		if groups is None:
			Xarr=X.reshape((1,NT,k))
			Yarr=Y.reshape((1,NT,1))
			NTW,k=W.shape
			Warr=W.reshape((1,NT,k))
			N=1
			max_T=NT
			T=np.array([[NT]])
		else:
			sel=np.unique(groups)
			N=len(sel)
			sel=(groups.T==sel.reshape((N,1)))
			T=np.sum(sel,1)
			max_T=np.max(T)
			Xarr=[]
			Yarr=[]
			Warr=[]
			k=0
			T_used=[]
			for i in sel:
				obs_count=np.sum(i)
				if obs_count>self.lost_obs+5:
					T_used.append(obs_count)
					Xarr.append(rp.fillmatr(X[i],max_T))
					Yarr.append(rp.fillmatr(Y[i],max_T))
					Warr.append(rp.fillmatr(W[i],max_T))
				else:
					print("Warning: group %s deleted because of too few observations" %(k))
				k+=1
			T=np.array(T_used)
			N=len(T)
		return np.array(Xarr),np.array(Yarr),np.array(Warr),max_T,T.reshape((N,1)),N

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and store all 
	associated dynamic variables needed outside this scope"""
	def __init__(self,args,panel,center_e=False):
		if args is None:
			args=panel.args.args
		self.LL_const=-0.5*np.pi*panel.NT_afterloss
		self.args_v=panel.args.conv_to_vector(panel,args)
		self.args_d=panel.args.conv_to_dict(panel,args)
		try:
			self.LL=self.LL_calc(panel,center_e)
		except Exception as e:
			self.LL=None
			#print(str(e))
		if not self.LL is None:
			if np.isnan(self.LL):
				self.LL=None
			
		
		
	def update(self,panel,args):
		self.args_v=panel.args.conv_to_vector(panel,args)
		self.args_d=panel.args.conv_to_dict(panel,args)
		self.LL=self.LL_calc(panel)
		

	def LL_calc(self,panel,center_e=False):
		args=self.args_d#using dictionary arguments
		matrices=panel.set_garch_arch(args,self)
		if matrices is None:
			return None		
		AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
		(N,T,k)=panel.X.shape

		u=panel.Y-fu.dot(panel.X,args['beta'])
		e=fu.dot(AMA_1AR,u)

		if panel.m>0:
			h_res=rp.h_func(e, args['z'][0])
			if h_res==None:
				return None
			(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val)=[i*panel.included for i in h_res]
			lnv_ARMA=fu.dot(GAR_1MA,h_val)
		else:
			(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val,avg_h)=(0,0,0,0,0,0,0)
			lnv_ARMA=0	
		W_omega=fu.dot(panel.W_a,args['omega'])
		lnv=W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
		if panel.m>0:
			avg_h=(np.sum(h_val,1)/panel.T_arr).reshape((N,1,1))*panel.a
			lnv=lnv+args['mu'][0]*avg_h
		v=np.exp(lnv)*panel.a
		v_inv=np.exp(-lnv)*panel.a	
		e_RE=rp.RE(self,panel,e)
		e_REsq=e_RE**2
		if center_e:
			e=e-np.mean(e)
		LL=self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
		if abs(LL)>1e+100: 
			return None
		self.AMA_1,self.AAR,self.AMA_1AR,self.GAR_1,self.GMA,self.GAR_1MA=matrices
		self.u,self.e,self.h_e_val,self.h_val, self.lnv_ARMA        = u,e,h_e_val,h_val, lnv_ARMA
		self.lnv,self.avg_h,self.v,self.v_inv,self.e_RE,self.e_REsq = lnv,avg_h,v,v_inv,e_RE,e_REsq
		self.h_2e_val,self.h_z_val,self.h_ez_val,self.h_2z_val      = h_2e_val,h_z_val,h_ez_val,h_2z_val
		self.e_st=e_RE*v_inv
		return LL

	def standardize(self,panel):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self"""
		v_inv=self.v_inv**0.5
		m=panel.lost_obs
		N,T,k=panel.X.shape
		Y=fu.dot(self.AMA_1AR,panel.Y)
		Y=rp.RE(self,panel,Y,False)*v_inv
		X=fu.dot(self.AMA_1AR,panel.X)
		X=rp.RE(self,panel,X,False)*v_inv
		self.e_st=self.e*v_inv
		self.Y_st=Y
		self.X_st=X
		self.e_st_long=panel.de_arrayize(self.e_st,m)
		self.Y_st_long=panel.de_arrayize(self.Y_st,m)
		self.X_st_long=panel.de_arrayize(self.X_st,m)

class arguments:
	"""Sets initial arguments and stores static properties of the arguments"""
	def __init__(self,p, d, q, m, k, panel, args,has_intercept):
		self.categories=['beta','rho','lambda','gamma','psi','omega','mu','z']
		self.set_init_args(p, d, q, m, k,panel, args,has_intercept)
		self.position_defs()

	def initargs(self,p,d,q,m,k,panel):
		args=dict()
		args['beta']=np.zeros((panel.X.shape[2],1))
		args['omega']=np.zeros((panel.W.shape[2],1))
		args['rho']=np.zeros(p)
		args['lambda']=np.zeros(q)
		args['psi']=np.zeros(m)
		args['gamma']=np.zeros(k)
		if m>0:
			args['mu']=np.array([1.0])
			args['z']=np.array([1.0])	
		else:
			args['mu']=np.array([])
			args['z']=np.array([])			
		return args

	def set_init_args(self,p,d,q,m,k,panel,args_old,has_intercept):
		
		args=self.initargs(p, d, q, m, k, panel)
		beta,e=stat.OLS(panel,panel.X,panel.Y,return_e=True)
		if args_old is None: 
			args['beta']=beta
			args['omega'][0][0]=np.log(np.var(e))
			if m>0:
				args['mu']=np.array([0.0])
				args['z']=np.array([1.0])		
		else:
			args['beta']=insert_arg(args['beta'],args_old['beta'])
			args['omega']=insert_arg(args['omega'],args_old['omega'])
			args['rho']=insert_arg(args['rho'],args_old['rho'])
			args['lambda']=insert_arg(args['lambda'],args_old['lambda'])
			args['psi']=insert_arg(args['psi'],args_old['psi'])
			args['gamma']=insert_arg(args['gamma'],args_old['gamma'])
			args['mu']=insert_arg(args['mu'],args_old['mu'])
			args['z']=insert_arg(args['z'],args_old['z'])
		self.args=args
		self.set_restricted_args(p, d, q, m, k,panel,e,beta)
		
	def new_args(self,beta,rho=[],lmbda=[],psi=[],gamma=[],omega=[],mu=[],z=[]):
		args=dict()
		args['beta']=np.array(beta)
		args['omega']=np.array(omega)
		args['rho']=np.array(rho)
		args['lambda']=np.array(lmbda)
		args['psi']=np.array(psi)
		args['gamma']=np.array(gamma)
		if m>0:
			args['mu']=np.array(mu)
			args['z']=np.array(z)	
		else:
			args['mu']=np.array(mu)
			args['z']=np.array(z)			
		return args

	def set_restricted_args(self,p, d, q, m, k, panel,e,beta):
		self.args_restricted=self.initargs(p, d, q, m, k, panel)
		self.args_OLS=self.initargs(p, d, q, m, k, panel)		
		self.args_restricted['beta'][0][0]=np.mean(panel.Y)
		self.args_restricted['omega'][0][0]=np.log(np.var(panel.Y))
		self.args_OLS['beta']=beta
		self.args_OLS['omega'][0][0]=np.log(np.var(e))
		
	def position_defs(self):
		"""Defines positions in vector argument"""

		self.positions=dict()
		self.map_to_categories=dict()
		k=0
		for i in self.categories:
			n=len(self.args[i])
			rng=range(k,k+n)
			self.positions[i]=rng
			for j in rng:
				self.map_to_categories[j]=i
			k+=n
	
	def conv_to_dict(self,panel,args):
		"""Converts a vector argument args to a dictionary argument. If args is a dict, it is returned unchanged"""
		if type(args)==dict:
			return args
		else:
			d=dict()
			k=0
			for i in panel.args.categories:
				n=len(panel.args.positions[i])
				rng=range(k,k+n)
				d[i]=args[rng]
				if i=='beta' or i=='omega':
					d[i]=d[i].reshape((n,1))
				k+=n
		return d


	def conv_to_vector(self,panel,args):
		"""Converts a dict argument args to vector argument. if args is a vector, it is returned unchanged"""
		if type(args)==dict:
			v=np.array([])
			for i in panel.args.categories:
				s=args[i]
				if type(s)==np.ndarray:
					s=s.flatten()

				v=np.concatenate((v,s))
			return v
		else:
			return args
		
			
def insert_arg(arg,add):
	n=min((len(arg),len(add)))
	arg[:n]=add[:n]
	return arg

class constraints:

	"""Stores the constraints of the LL maximization"""
	def __init__(self,args,old_constr,add_one_constr):
		self.constraints=dict()
		self.categories=[]
		self.args=args
		if old_constr is None:
			self.old_constr=[]
		else:
			self.old_constr=old_constr.constraints
		self.add_one_constr=add_one_constr

	def add(self,positions, minimum_or_value,maximum=None,replace=True):
		"""Adds a constraint. 'positions' is either an integer or an iterable of integer specifying the position(s) 
		for which the constraints shall apply. If 'positions' is a string, it is assumed to be the name of a category \n\n
		
		Equality constraints are chosen by specifying 'value' \n\n
		Inequality constraints are chosen specifiying 'maximum' and 'minimum'\n\n
		'replace' determines whether an existing constraint shall be replaced or not 
		(only one equality and inequality allowed per position)"""
		if type(positions)==int or type(positions)==np.int64  or type(positions)==np.int32:
			positions=[positions]
		elif type(positions)==str:
			positions=self.args.positions[positions]
		for i in positions:
			if replace or (i not in self.constraints):
				if maximum==None:
					self.constraints[i]=[minimum_or_value]
				else:
					if minimum_or_value<maximum:
						self.constraints[i]=[minimum_or_value,maximum]
					else:
						self.constraints[i]=[maximum,minimum_or_value]
			category=self.args.map_to_categories[i]
			if not category in self.categories:
				self.categories.append(category)
			

	def constraints_to_arrays(self):
		c=[]
		c_eq=[]
		for i in self.constraints:
			if len(self.constraints[i])==1:
				c_eq.append(self.constraints[i]+[i])
			else:
				c.append(self.constraints[i]+[i])
		return c,c_eq
	
	def remove(self):
		"""Removes arbitrary constraint"""
		k=list(self.constraints.keys())[0]
		self.constraints.pop(k)
		
				





