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
import paneltime_functions as ptf
import calculus
import copy
import loglikelihood as logl
from scipy import sparse as sp


min_AC=0.000001


def posdef(a,da):
	return list(range(a,a+da)),a+da

class panel:
	def __init__(self,p,d,q,m,k,X,Y,groups,x_names,y_name,groups_name,fixed_random_eff,W,w_names,descr,data,h,has_intercept,loadargs,model_string,user_constraints):
		"""
		No effects    : fixed_random_eff=0\n
		Fixed effects : fixed_random_eff=1\n
		Random effects: fixed_random_eff=2\n
		
		"""
		if groups_name is None:
			fixed_random_eff=0

		self.initial_defs(h,X,Y,groups,W,has_intercept,data,p,q,m,k,d,x_names,y_name,groups_name,w_names,descr,fixed_random_eff,loadargs,model_string)
		
		self.X,self.Y,self.W,self.max_T,self.T_arr,self.N=self.arrayize(X, Y, W, groups)

		self.masking()
		self.lag_variables(max((q,p,k,m)))
		

		self.final_defs(p,d,q,m,k,X,user_constraints)

		self.between_group_ols()

	def masking(self):
		self.date_counter=np.arange(self.max_T).reshape((self.max_T,1))
		#"initial observations" mask: 
		self.a=np.array([self.date_counter<self.T_arr[i] for i in range(self.N)])# sets observations that shall be zero to zero by multiplying it with the arrayized variable
	
		#"after lost observations" masks: 
		self.included=np.array([(self.date_counter>=self.lost_obs)*(self.date_counter<self.T_arr[i]) for i in range(self.N)])# sets observations that shall be zero after lost observations to zero by multiplying it with the arrayized variable
		self.n_i=np.sum(self.included,1).reshape((self.N,1,1))#number of observations for each i
		self.n_i=self.n_i+(self.n_i<=0)#ensures minimum of 1 observation in order to avoid division error. If there are no observations, averages will be zero in any case	
		
	def initial_defs(self,h,X,Y,groups,W,has_intercept,data,p,q,m,k,d,x_names,y_name,groups_name,w_names,descr,fixed_random_eff,loadargs,model_string):
		self.args_bank=ptf.args_bank(model_string, loadargs)		
		self.loadargs=loadargs
		self.has_intercept=has_intercept

		self.data=data
		self.lost_obs=np.max((p,q))+max((m,k))+d#+3
		self.x_names=x_names
		self.y_name=y_name
		self.raw_X=X
		self.raw_Y=Y
		self.groups_name=groups_name
		self.w_names=w_names		
		self.p,self.d,self.q,self.m,self.k,self.nW,self.n_beta=p,d,q,m,k,W.shape[1],X.shape[1]
		self.n_beta=len(X[0])
		self.descr=descr
		self.its_reg=0
		self.FE_RE=fixed_random_eff
		self.groups=groups	
		self.len_data=len(X)
		self.define_h_func(h)
		
		
		
	def final_defs(self,p,d,q,m,k,X,user_constraints):
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
		self.args=arguments(p, d, q, m, k, self, self.args_bank.args,self.has_intercept,user_constraints)



	
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
		T=self.max_T
		self.I=np.diag(np.ones(T))
		self.zero=self.I*0
		
		#for sparse matrices:
		self.cnt_sp=np.arange(T)
		self.ones_sp=np.ones(T)
		self.I_sp=sp.csc_matrix((self.ones_sp,(self.cnt_sp,self.cnt_sp)),(T,T))
		self.L_sp=[sp.csc_matrix((self.ones_sp[i+1:],(self.cnt_sp[i+1:],self.cnt_sp[:-i-1])),(T,T)) for i in range(max_lags)]			

		#differencing:
		if self.d==0:
			return
		L0=np.diag(np.ones(T-1),-1)
		Ld=(self.I-L0)
		for i in range(1,self.d):
			Ld=fu.dot(self.I-L0,self.Ld)		
		self.Y=fu.dot(Ld,self.Y)*self.a	
		self.X=fu.dot(Ld,self.X)*self.a
		if self.has_intercept:
			self.X[:,:,0]=1
		self.Y[:,:self.d]=0
		self.X[:,:self.d]=0	

	def params_ok(self,args):
		a=self.q_sel,self.p_sel,self.M_sel,self.K_sel
		for i in a:
			if len(i)>0:
				if np.any(np.abs(args[i])>0.999):
					return False
		return True



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
					print("Warning: group %s removed because of too few observations" %(k))
				k+=1
			T=np.array(T_used)
			N=len(T)
		return np.array(Xarr),np.array(Yarr),np.array(Warr),max_T,T.reshape((N,1)),N
	
	def define_h_func(self,h_definition):
		global h
		if h_definition is None:
			h_definition="""
	def h(e,z):
		ez2=e**2+z**2+1e-15
		h_val		=	 np.log(ez2)	
		h_e_val		=	 2*e/ez2
		h_2e_val	=	 2*(z**2-e**2)/(ez2**2)
		h_z_val		=	 2*z/ez2
		h_2z_val	=	2*(e**2-z**2)/(ez2**2)
		h_ez_val	=	-4*e*z/ez2**2
		return h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val
	"""	
		n=h_definition.find('def h(e,z)')
		for i in ['h(e,z)']:
			n=h_definition.find(i)
		
		h_definition=h_definition.replace('\t','    ')
		h_list=h_definition.split('\n')
		k=len(h_list)
		for i in range(k):
			n=h_list[i].find('def h(')
			if n>0:
				h_list[i]=h_list[i][:n+5]+h_list[i][n+5:].replace(' ','')
				n=h_list[i].find('def h(e,z)')
			if n>-1:
				break
		if n<0:
			raise RuntimeError('The h-function must be defined as  "def h(e,z)..."')
		for i in range(len(h_list)):
			h_list[i]=h_list[i][n:]
		h_definition='\n'.join(h_list)
		try:
			exec(h_definition)
		except IndentationError:
			raise RuntimeError("""IndentationError in your custom h-function: 
		    You might have a mix of spaces and tabs. 
		    Space indentation only is recommended. 
		    Tab-size four is assumed""")
		s='ret=h(e,z)'
		s='\n\n'+s
		self.h_def=h_definition+s

class arguments:
	"""Sets initial arguments and stores static properties of the arguments"""
	def __init__(self,p, d, q, m, k, panel, args,has_intercept,user_constraints):
		self.categories=['beta','rho','lambda','gamma','psi','omega','mu','z']
		self.set_init_args(p, d, q, m, k,panel, args,has_intercept,user_constraints)
		self.make_namevector(panel,p, q, m, k)
		self.position_defs()
		N,T,b=panel.X.shape
		self.n_args=b+p+q+m+k+(m>0)*2+panel.W.shape[2]-(N==1)
		self.apply_user_constraints(panel, user_constraints)

	def initargs(self,p,d,q,m,k,panel):
		args=dict()
		args['beta']=np.zeros((panel.X.shape[2],1))
		args['omega']=np.zeros((panel.W.shape[2],1))
		args['rho']=np.zeros(p)
		args['lambda']=np.zeros(q)
		args['psi']=np.zeros(m)
		args['gamma']=np.zeros(k)
		args['mu']=np.array([])
		args['z']=np.array([])	
		if m>0 and panel.N>1:
			args['mu']=np.array([1.0])
		if m>0:
			args['z']=np.array([1.0])	

		return args

	def set_init_args(self,p,d,q,m,k,panel,args_old,has_intercept,user_constraints):
		
		args=self.initargs(p, d, q, m, k, panel)
		
		#de2=np.roll(e**2,1)-e**2
		#c=stat.correl(np.concatenate((np.roll(de2,1),de2),2),panel)[0,1]

		beta,e=stat.OLS(panel,panel.X,panel.Y,return_e=True)
		args['beta']=beta
		args['omega'][0][0]=np.log(np.var(e*panel.included)*len(e[0])/np.sum(panel.included))
		if m>0:
			if panel.N>1:
				args['mu']=np.array([0.0])
			args['z']=np.array([1.0])	
		self.start_args=fu.copy_array_dict(args)
		if not args_old is None: 
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
		

	def set_restricted_args(self,p, d, q, m, k, panel,e,beta):
		self.args_restricted=self.initargs(p, d, q, m, k, panel)
		self.args_OLS=self.initargs(p, d, q, m, k, panel)		
		self.args_restricted['beta'][0][0]=np.mean(panel.Y)
		self.args_restricted['omega'][0][0]=np.log(np.var(panel.Y))
		self.args_OLS['beta']=beta
		self.args_OLS['omega'][0][0]=np.log((np.var(e*panel.included)*len(e[0])/np.sum(panel.included)))
		
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
	
	def conv_to_dict(self,args):
		"""Converts a vector argument args to a dictionary argument. If args is a dict, it is returned unchanged"""
		if type(args)==dict:
			return args
		else:
			d=dict()
			k=0
			for i in self.categories:
				n=len(self.positions[i])
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


	def make_namevector(self,panel,p, q, m, k):
		"""Creates a vector of the names of all regression varaibles, including variables, ARIMA and GARCH terms"""
		d=dict()
		names=panel.x_names[:]#copy variable names
		d['beta']=names
		add_names(p,'AR term %s (p)','rho',d,names)
		add_names(q,'MA term %s (q)','lambda',d,names)
		add_names(m,'MACH term %s (m)','gamma',d,names)
		add_names(k,'ARCH term %s (k)','psi',d,names)
		d['omega']=panel.w_names
		names.extend(panel.w_names)
		if m>0:
			if panel.N>1:
				d['mu']=['mu (var.group eff.)']
				names.extend(d['mu'])
			d['z']=['z in h(e,z)']
			names.extend(d['z'])
			
		self.names_v=names
		self.names_d=d
		
	def apply_user_constraints(self,panel,user_constraints):
		if user_constraints is None:
			return	
		if type(user_constraints)!=list:
			print("Warning: user user_constraints must be a list of tuples. user_constraints are not applied.")	
		names_v=self.names_v
		names_d=self.names_d
		if type(user_constraints[0])!=list:
			user_constraints=[user_constraints]	
		for i in user_constraints:
			if len(i)==2:
				if type(i[0])!=str:
					name=names[i[0]]
				else:
					name=i[0]
			for c in self.categories:
				if name in names_d[c]:
					j=names_d[c].index(name)
					self.start_args[c][j]=i[1]
					self.args[c][j]=i[1]
					self.args_restricted[c][j]=i[1]
					self.args_OLS[c][j]=i[1]
					break
			

			
def add_names(T,namsestr,category,d,names):
	a=[]
	for i in range(T):
		a.append(namsestr %(i,))
	names.extend(a)
	d[category]=a
	

def insert_arg(arg,add):
	n=min((len(arg),len(add)))
	arg[:n]=add[:n]
	return arg






