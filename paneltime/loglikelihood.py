#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object

import numpy as np
import functions as fu
import regprocs as rp
import statproc as stat
import calculus

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and store all 
	associated dynamic variables needed outside this scope"""
	def __init__(self,args,panel):
		if args is None:
			args=panel.args.args
		self.LL_const=-0.5*np.log(2*np.pi)*panel.NT_afterloss
		self.args_v=panel.args.conv_to_vector(panel,args)
		self.args_d=panel.args.conv_to_dict(args)
		self.LL=self.LL_calc(panel)
		try:
			self.LL=self.LL_calc(panel)
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


	def LL_calc(self,panel):
		args=self.args_d#using dictionary arguments


		matrices=set_garch_arch(panel,args)

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
			if panel.N>1:
				lnv=lnv+args['mu'][0]*avg_h
			lnv=np.maximum(np.minimum(lnv,100),-100)
		v=np.exp(lnv)*panel.a
		v_inv=np.exp(-lnv)*panel.a	
		e_RE=rp.RE(self,panel,e)
		e_REsq=e_RE**2
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
		self.e_st=self.e_RE*v_inv
		self.Y_st=Y
		self.X_st=X
		self.e_st_long=panel.de_arrayize(self.e_st,m)
		self.Y_st_long=panel.de_arrayize(self.Y_st,m)
		self.X_st_long=panel.de_arrayize(self.X_st,m)

	def copy_args_d(self):
		return fu.copy_array_dict(self.args_d)
	
	


def set_garch_arch(panel,args):
	if panel.max_T>50000:#For really large matrices, sparse inversion is faster and less memory consuming
		return set_garch_arch_sparse(panel,args)

	p,q,m,k,nW=panel.p,panel.q,panel.m,panel.k,panel.nW
	X=panel.I+lag_matr(panel.L,panel.zero,q,args['lambda'])
	try:
		AMA_1=np.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(AMA_1)):
		return None
	AAR=panel.I-lag_matr(panel.L,panel.zero,p,args['rho'])
	AMA_1AR=fu.dot(AMA_1,AAR)
	X=panel.I-lag_matr(panel.L,panel.zero,k,args['gamma'])
	try:
		GAR_1=np.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(GAR_1)):
		return None		
	GMA=lag_matr(panel.L,panel.zero,m,args['psi'])	
	GAR_1MA=fu.dot(GAR_1,GMA)
	return AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA



def set_garch_arch_sparse(panel,args):
	p,q,m,k,nW=panel.p,panel.q,panel.m,panel.k,panel.nW

	X=panel.I_sp+lag_matr_sp(q,args['lambda'])
	try:
		AMA_1=sp.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(AMA_1.data)):
		return None
	AAR=panel.I_sp-lag_matr_sp(p,args['rho'])
	AMA_1AR=fu.dot(AMA_1,AAR)
	X=panel.I_sp-lag_matr_sp(k,args['gamma'])
	try:
		GAR_1=sp.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(GAR_1.data)):
		return None		
	GMA=lag_matr_sp(m,args['psi'])
	GAR_1MA=fu.dot(GAR_1,GMA)

	return AMA_1.toarray(),AAR.toarray(),AMA_1AR.toarray(),GAR_1.toarray(),GMA.toarray(),GAR_1MA.toarray()



def lag_matr_sp(panel,k,args):
	T=panel.max_T
	M=sp.csc_matrix(([],([],[])),(T,T))
	for i in range(k):
		M=M+args[i]*panel.L_sp[i]
	return M


def lag_matr(L,zero,k,args):
	if k==0:
		return zero
	a=[]
	for i in range(k):
		a.append(args[i]*L[i])
	a=np.sum(a,0)
	return a


class direction:
	def __init__(self,panel):
		self.gradient=calculus.gradient(panel)
		self.hessian=calculus.hessian(panel)
		self.panel=panel
		self.constr=None
		
		
	def get(self,ll,mc_limit,add_one_constr,dx_conv,k,its,H_old,mp=None):

		g,G=self.gradient.get(ll,return_G=True)
		hessian=self.hessian.get(ll,mp)

		dc,constrained,reset,out,constr=self.solve(add_one_constr, G, g, hessian, ll, mc_limit, 
		                                           dx_conv,k,its)

		#fixing positive definit hessian (convexity problem) by using robust sandwich estimator

		if np.sum(dc*(constrained==0)*g)<0:
			#print("Warning: negative slope. Using robust sandwich hessian matrix to ensure positivity")
			hessin=rp.sandwich(hessian,G,0)
			for i in range(len(hessin)):
				hessin[i,i]=hessin[i,i]+(hessin[i,i]==0)
			hessian=-np.linalg.inv(hessin)
			dc,constrained,reset,out,constr=self.solve(add_one_constr, G, g, hessian, ll, mc_limit, 
			                                           dx_conv,k,its)			

		if len(constr.constraints)>0:
			out.print()		
		return dc,g,G,hessian,constrained,reset

	def solve(self,add_one_constr,G,g,hessian,ll,mc_limit,dx_conv,k,its):	
		constr=constraints(self.panel.args,self.constr,add_one_constr)
		self.constr=constr
		hessian,reset,out=add_constraints(G,self.panel,ll,constr,mc_limit,dx_conv,hessian,k,its)
		dc,constrained=solve(constr,hessian, g, ll.args_v)
		return dc,constrained,reset,out,constr
	

def solve(constr,H, g, x):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	n=len(H)
	c,c_eq=constr.constraints_to_arrays()
	k=len(c)
	m=len(c_eq)
	H=np.concatenate((H,np.zeros((n,k+m))),1)
	H=np.concatenate((H,np.zeros((k+m,n+k+m))),0)
	g=np.append(g,(k+m)*[0])


	r_eq_indicies=[]
	for i in range(k+m):
		H[n+i,n+i]=1
	for i in range(m):
		j=int(c_eq[i][1])
		H[j,n+i]=1
		H[n+i,j]=1
		H[n+i,n+i]=0
		g[n+i]=-(c_eq[i][0]-x[j])
		r_eq_indicies.append(j)
	sel=[i for i in range(len(H))]
	H[sel,sel]=H[sel,sel]+(H[sel,sel]==0)*1e-15
	xi=-np.linalg.solve(H,g).flatten()
	for i in range(k):#Kuhn-Tucker:
		j=int(c[i][2])
		q=None
		if j in r_eq_indicies:
			q=None
		elif x[j]+xi[j]<c[i][0]-1e-15:
			q=-(c[i][0]-x[j])
		elif x[j]+xi[j]>c[i][1]+1e-15:
			q=-(c[i][1]-x[j])
		if q!=None:
			H[j,n+i+m]=1
			H[n+i+m,j]=1
			H[n+i+m,n+i+m]=0
			g[n+i+m]=q
			xi=-np.linalg.solve(H,g).flatten()	
	constrained=np.sum(H[n:,:n],0)
	return xi[:n],constrained

def remove_constants(panel,G,include,constr,out,names):
	N,T,k=G.shape
	v=stat.var(panel,G)
	for i in range(1,k):
		if v[0][i]==0:
			include[i]=False
			constr.add(i,0)
			out.add(names[i],0,'NA','constant')	


def remove_H_correl(hessian,include,constr,args,out,names):
	k,k=hessian.shape
	hessian_abs=np.abs(hessian)
	x=(np.diag(hessian_abs)**0.5).reshape((1,k))
	x=(x.T*x)
	corr=hessian_abs/(x+(x==0)*1e-100)	
	for i in range(k):
		m=np.max(corr[i])
		if m>2*corr[i,i]:
			j=np.nonzero(corr[i]==m)[0][0]
			corr[:,j]=0
			corr[j,:]=0
			corr[j,j]=1	
	for i in range(k):
		corr[i,i:]=0

	p=np.arange(k).reshape((1,k))*np.ones((k,1))
	p=np.concatenate((corr.reshape((k,k,1)),p.T.reshape((k,k,1)),p.reshape((k,k,1))),2)
	p=p.reshape((k*k,3))
	srt=np.argsort(p[:,0],0)
	p=p[srt][::-1]
	p=p[np.nonzero(p[:,0]>=1.0)[0]]
	principal_factors=[]
	groups=correl_groups(p)
	acc=None
	for i in groups:
		for j in range(len(i)):
			if not i[j] in constr.constraints:
				acc=i.pop(j)
				break
		if not acc is None:
			for j in i:
				remvd=remove(j,acc,args,include,out,constr,names,'h-correl')	
	return hessian

def remove_correl(panel,G,include,constr,args,out,names):
	N,T,k=G.shape
	corr=np.abs(stat.correl(G,panel))
	for i in range(k):
		corr[i,i:]=0

	p=np.arange(k).reshape((1,k))*np.ones((k,1))
	p=np.concatenate((corr.reshape((k,k,1)),p.T.reshape((k,k,1)),p.reshape((k,k,1))),2)
	p=p.reshape((k*k,3))
	srt=np.argsort(p[:,0],0)
	p=p[srt][::-1]
	p=p[np.nonzero(p[:,0]>0.8)[0]]
	principal_factors=[]
	groups=correl_groups(p)
	for i in groups:
		for j in range(len(i)):
			if not i[j] in constr.constraints:
				acc=i.pop(j)
				break
		for j in i:
			remvd=remove(j,acc,args,include,out,constr,names,'correl')	


def append_to_group(group,intlist):
	ingroup=False
	for i in intlist:
		if i in group:
			ingroup=True
			break
	if ingroup:
		for j in intlist:
			if not j in group:
				group.append(j)
		return True
	else:
		return False

def correl_groups(p):
	groups=[]
	appended=False
	x=np.array(p[:,1:3],dtype=int)
	for i,j in x:
		for k in range(len(groups)):
			appended=append_to_group(groups[k],[i,j])
			if appended:
				break
		if not appended:
			groups.append([i,j])
	g=len(groups)
	keep=g*[True]
	for k in range(g):
		if keep[k]:
			for h in range(k+1,len(groups)):
				appended=False
				for m in range(len(groups[h])):
					if groups[h][m] in groups[k]:
						appended=append_to_group(groups[k],  groups[h])
						keep[h]=False
						break
	g=[]
	for i in range(len(groups)):
		if keep[i]:
			g.append(groups[i])
	return g


def remove_one_multicoll(G,args,names,include,out,constr,limit):
	n=len(include)
	T,N,k=G.shape
	c_index,var_prop=stat.var_decomposition(X=G[:,:,include])
	zeros=np.zeros(len(c_index))
	c_index=c_index.flatten()
	for i in range(k):
		if not include[i]:
			c_index=np.insert(c_index,i,0)
			var_prop=np.insert(var_prop,i,zeros,1)

	if c_index[-1]>limit:
		if np.sum(var_prop[-1]>0.5)>1:
			j=np.argsort(var_prop[-1])[-1]
			assc=np.argsort(var_prop[-1])[-2]
			remvd=remove(j, assc,args, include, out,constr,names,'collinear')
			return True
	return False

def remove_all_multicoll(G,args,names,include,out,constr,limit):
	T,N,k=G.shape
	for i in range(k):
		remvd=remove_one_multicoll(G,args,names,include,out,constr,limit)
		if not remvd:
			return


def remove(d,assoc,args,include,out,constr,names,r_type):
	if d in constr.constraints:
		return False

	if type(args)==list or type(args)==np.ndarray:
		args=args[d]
	constr.add(d,args)
	include[d]=False	
	if not assoc is None:
		out.add(names[d],args,names[assoc],r_type)	
	else:
		out.add(names[d],args,'NA',r_type)	
	return True

def add_constraints(G,panel,ll,constr,mc_limit,dx_conv,hessian,k,its):
	names=panel.name_vector
	args=ll.args_v
	N,T,h=G.shape
	include=np.ones(h,dtype=bool)
	out=output()
	add_initial_constraints(panel,constr,out,names,ll,include,its)
	remove_constants(panel, G, include,constr,out,names)	
	remove_all_multicoll(G, args, names, include, out, constr, 5000)
	reset=False
	#remove_H_correl(hessian,include,constr,args,out,names)
	if mc_limit<30 and not (dx_conv is None):
		srt=np.argsort(dx_conv)
		for i in range(min((k,len(srt)-2))):
			j=srt[-i-1]
			if dx_conv[j]<0.05:
				reset=True
			else:
				reset=remove(j,None,args, include, out,constr,names,'dir cap')==False
	return hessian, reset,out

def add_initial_constraints(panel,constr,out,names,ll,include,its):
	args=panel.args
	if its<-3:
		for a in ['beta','rho','gamma','psi','lambda']:
			for i in args.positions[a][1:]:
				remove(i, None, 0, include, out, constr, names, 'initial')
	if its==-3:
		ll.standardize(panel)
		beta=stat.OLS(panel,ll.X_st,ll.Y_st)
		for i in args.positions['beta']:
			remove(i, None, beta[i][0], include, out, constr, names, 'initial')	




class output:
	def __init__(self):
		self.variable=[]
		self.set_to=[]
		self.assco=[]
		self.cause=[]

	def add(self,variable,set_to,assco,cause):
		if (not (variable in self.variable)) or (not (cause in self.cause)):
			self.variable.append(variable)
			self.set_to.append(str(round(set_to,8)))
			self.assco.append(assco)
			self.cause.append(cause)

	def print(self):
		output= "|Restricted variable |    Set to    | Associated variable|  Cause   |\n"
		output+="|--------------------|--------------|--------------------|----------|\n"
		if len(self.variable)==0:
			return
		for i in range(len(self.variable)):
			output+="|%s|%s|%s|%s|\n" %(
		        self.variable[i].ljust(20)[:20],
		        self.set_to[i].rjust(14)[:14],
		        self.assco[i].ljust(20)[:20],
		        self.cause[i].ljust(10)[:10])	

		print(output)	


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

		Equality constraints are chosen by specifying 'minimum_or_value' \n\n
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



