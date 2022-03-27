#!/usr/bin/env python
# -*- coding: utf-8 -*-
import calculus
import calculus_functions as cf
import calculus_ll as cll
import numpy as np
import constraints
import loglikelihood as logl
from scipy import stats
import sys
import time
import maximize_num


class direction:
	def __init__(self,panel,mp,channel):
		self.progress_bar=channel.set_progress
		self.gradient=calculus.gradient(panel,self.progress_bar)
		self.hessian=calculus.hessian(panel,self.gradient,self.progress_bar)
		self.panel=panel
		self.constr=None
		self.hessian_num=None
		self.do_shocks=True
		self.input_old=None
		self.CI=0
		self.mp=mp
		self.dx_norm=None
		self.dx=None
		self.H=None
		self.G=None
		self.g=None
		self.weak_mc_dict=[]
		self.mc_problems=[]
		self.H_correl_problem=False
		self.singularity_problems=False
		self.record=[]
		self.Hess_correction=1
		self.g_rec=[]
		self.start_time=time.time()


	def calculate(self,ll,its,msg,increment,lmbda,rev):
		if ll.LL is None:
			raise RuntimeError("Error in LL calculation: %s" %(ll.err_msg,))
		x_old=self.ll.args.args_v
		self.ll=ll
		self.xi=ll.args.args_v-x_old
		self.its=its
		self.lmbda=lmbda
		self.has_reversed_directions=rev
		self.increment=increment
		self.constr_old=self.constr
		#self.constr=constraints.constraints(self,ll.args,its)
		#self.constr.add_static_constraints()			
		self.calc_gradient()
		self.calc_hessian()
		if lmbda<0.05 and False:
			self.constr.add_dynamic_constraints(self)	
			self.CI=self.constr.CI
		if self.constr is None:
			self.H_correl_problem,self.mc_problems,self.weak_mc_dict=False, [],{}
		else:
			self.H_correl_problem=self.constr.H_correl_problem	
			self.mc_problems=self.constr.mc_problems
			self.weak_mc_dict=self.constr.weak_mc_dict
		
		self.singularity_problems=(len(self.mc_problems)>0) or self.H_correl_problem
		
	def set(self,mp,args,constraints=None):
		if not constraints is None:
			self.constr.add_custom_constraints(constraints,
										 cause='pedantic constraint',
										 clear=True,args=args)
		if not mp is None:
			mp.send_dict_by_file({'constr':self.constr})
		self.constr=None
		self.dx=solve(self.H, self.g, args.args_v, self.constr)
		#self.dx=solve_delete(self.constr,self.H, self.g, args.args_v)	
		self.dx_norm=self.normalize(self.dx)

			
	
	def include(self,all=False):
		include=np.array(self.panel.args.n_args*[True])
		if all:
			return include
		include[list(self.constr)]=False
		return include	

	def calc_gradient(self,ll=None):
		if ll is None:
			ll=self.ll
		dLL_lnv, DLL_e=cll.gradient(ll,self.panel)
		self.LL_gradient_tobit(ll, DLL_e, dLL_lnv)
		self.g_old=self.g
		self.g,self.G=self.gradient.get(ll,DLL_e,dLL_lnv,return_G=True)
	
	def LL_gradient_tobit(self,ll,DLL_e,dLL_lnv):
		g=[1,-1]
		self.f=[None,None]
		self.f_F=[None,None]
		for i in [0,1]:
			if self.panel.tobit_active[i]:
				I=self.panel.tobit_I[i]
				self.f[i]=stats.norm.pdf(g[i]*ll.e_norm[I])
				self.f_F[i]=(ll.F[i]!=0)*self.f[i]/(ll.F[i]+(ll.F[i]==0))
				self.v_inv05=ll.v_inv**0.5
				DLL_e[I]=g[i]*self.f_F[i]*self.v_inv05[I]
				dLL_lnv[I]=-0.5*DLL_e[I]*ll.e_RE[I]
				a=0
				

	def LL_hessian_tobit(self,ll,d2LL_de2,d2LL_dln_de,d2LL_dln2):
		g=[1,-1]
		if sum(self.panel.tobit_active)==0:
			return
		self.f=[None,None]
		e1s1=ll.e_norm
		e2s2=ll.e_REsq*ll.v_inv
		e3s3=e2s2*e1s1
		e1s2=e1s1*self.v_inv05
		e1s3=e1s1*ll.v_inv
		e2s3=e2s2*self.v_inv05
		f_F=self.f_F
		for i in [0,1]:
			if self.panel.tobit_active[i]:
				I=self.panel.tobit_I[i]
				f_F2=self.f_F[i]**2
				d2LL_de2[I]=      -(g[i]*f_F[i]*e1s3[I] + f_F2*ll.v_inv[I])
				d2LL_dln_de[I] =   0.5*(f_F2*e1s2[I]  +  g[i]*f_F[i]*(e2s3[I]-self.v_inv05[I]))
				d2LL_dln2[I] =     0.25*(f_F2*e2s2[I]  +  g[i]*f_F[i]*(e1s1[I]-e3s3[I]))
				
				
	def nummerical_hess(self):
		hessin=inv_hess(self.H)
		hessin=nummerical_hessin(self.g,self.g_old,hessin,self.dx)
		return inv_hess(hessin)		

	def calc_hessian(self):
		d2LL_de2, d2LL_dln_de, d2LL_dln2 = cll.hessian(self.ll,self.panel)
			
		self.LL_hessian_tobit(self.ll, d2LL_de2, d2LL_dln_de, d2LL_dln2)
		if self.H is None:
			self.H = -np.identity(len(self.ll.args.args_v))
		else:
			hessin=np.linalg.inv(self.H)
			hessin = -maximize_num.hessin_num(hessin, -self.g, -self.g_old, self.xi)
			self.H = np.linalg.inv(hessin)
			#self.H = self.nummerical_hess()
		return
		if False:
			if not self.H is None:
				H=self.nummerical_hess()
			if H is None:# or self.increment<1:#(self.its/2==int(self.its/2)):
				self.H=self.hessian.get(self.ll,self.mp,d2LL_de2,d2LL_dln_de,d2LL_dln2)
			else:
				self.H=H
		d=np.diag(self.H)
		self.CI=constraints.decomposition(self.H)[0][-1]
		det=np.linalg.det(self.H)
		self.record.append([self.ll.LL,self.increment,self.CI,det]+list(d))

		if (np.any(d==0) 
			#or (np.any(d>=0) and det>0) 
			or (self.CI>1e+18 and self.increment<1)
			or self.increment<1e-3
			):
			self.H=0.5*(self.H+np.diag(d))
			self.progress_bar(text='Hessian diagonal doubled because it had positive or zero elements')

	
	def init_ll(self,args):
		self.constr=constraints.constraints(self,args)
		self.constr.add_static_constraints()			
		try:
			args=args.args_v
		except:
			pass#args must be a vector
		for i in self.constr.fixed:
			args[i]=self.constr.fixed[i].value
		#ll=logl.LL(args, self.panel, constraints=self.constr,print_err=True)
		#testing:
		self.function=maximize_num.Function(args, self.panel, self.constr)
		ll=self.function.ll
		#-----
		if ll.LL is None:
			if self.panel.options.loadargs.value:
				print("WARNING: Initial arguments failed, attempting default OLS-arguments ...")
				self.panel.args.set_init_args(self.panel,default=True)
				ll=logl.LL(self.panel.args.args_OLS,self.panel,constraints=self.constr,print_err=True)
				if ll.LL is None:
					raise RuntimeError("OLS-arguments failed too, you should check the data")
				else:
					print("default OLS-arguments worked")
			else:
				raise RuntimeError("OLS-arguments failed, you should check the data")
		self.ll=ll
		return ll
	
	def normalize(self,dx):
		args_v=self.ll.args.args_v
		dx_norm=(args_v!=0)*dx/(np.abs(args_v)+(args_v==0))
		dx_norm=(args_v<1e-2)*dx+(args_v>=1e-2)*dx_norm	
		return dx_norm	
	

def inv_hess(hessian):
	try:
		h=-np.linalg.inv(hessian)
	except:
		return None	
	return h

def nummerical_hessin(g,g_old,hessin,dxi):
	
	if g is None or g_old is None or hessin is None or dxi is None:
		return None
	dg=g-g_old 				#Compute difference of gradients,
	#and difference times current matrix:
	n=len(g)
	hdg=(np.dot(hessin,dg.reshape(n,1))).flatten()
	fac=fae=sumdg=sumxi=0.0 							#Calculate dot products for the denominators. 
	fac = np.sum(dg*dxi) 
	fae = np.sum(dg*hdg)
	sumdg = np.sum(dg*dg) 
	sumxi = np.sum(dxi*dxi) 
	if (fac > (3.0e-16*sumdg*sumxi)**0.5):#Skip update if fac not sufficiently positive.
		fac=1.0/fac
		fad=1.0/fae 
								#The vector that makes BFGS different from DFP:
		dg=fac*dxi-fad*hdg   
		#The BFGS updating formula:
		hessin+=fac*dxi.reshape(n,1)*dxi.reshape(1,n)
		hessin-=fad*hdg.reshape(n,1)*hdg.reshape(1,n)
		hessin+=fae*dg.reshape(n,1)*dg.reshape(1,n)	
	return hessin


def solve(H, g, x, constr):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	if H is None:
		raise RuntimeError('Hessian is None')
	dx_init=-np.linalg.solve(H,g).flatten()	
	if constr is None:
		return dx_init	
	n=len(H)
	k=len(constr)
	H=np.concatenate((H,np.zeros((n,k))),1)
	H=np.concatenate((H,np.zeros((k,n+k))),0)
	g=np.append(g,(k)*[0])
	
	for i in range(k):
		H[n+i,n+i]=1
	j=0
	dx=np.zeros(len(g))
	for i in constr.fixed:
		#kuhn_tucker(constr.fixed[i],i,j,n, H, g, x,dx, recalc=False)
		kuhn_tucker2(constr.fixed[i],i,j,n, H, g, x,dx, dx_init, recalc=False)
		j+=1
	dx=-np.linalg.solve(H,g).flatten()	
	OK=False
	w=0
	for r in range(50):
		j2=j
		
		for i in constr.intervals:
			dx=kuhn_tucker(constr.intervals[i],i,j2,n, H, g, x,dx)
			j2+=1
		OK=constr.within(x+dx[:n],False)
		if OK: 
			break
		if r==k+3:
			#print('Unable to set constraints in direction calculation')
			break

	return dx[:n]


def solve_delete(constr,H, g, x):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	if H is None:
		return None,g*0
	try:
		list(constr.keys())[0]
	except:
		return -np.linalg.solve(H,g).flatten()	
	
	m=len(H)
	
	idx=np.ones(m,dtype=bool)
	delmap=np.arange(m)
	if len(list(constr.fixed.keys()))>0:#removing fixed constraints from the matrix
		idx[list(constr.fixed.keys())]=False
		H=H[idx][:,idx]
		g=g[idx]
		delmap-=np.cumsum(idx==False)
		delmap[idx==False]=m#if for some odd reason, the deleted variables are referenced later, an out-of-bounds error is thrown
	n=len(H)
	k=len(constr.intervals)
	H=np.concatenate((H,np.zeros((n,k))),1)
	H=np.concatenate((H,np.zeros((k,n+k))),0)
	g=np.append(g,(k)*[0])
	
	for i in range(k):
		H[n+i,n+i]=1
	dx=-np.linalg.solve(H,g).flatten()	
	xi_full=np.zeros(m)
	OK=False
	keys=list(constr.intervals.keys())
	for r in range(50):		
		for j in range(k):
			key=keys[j]
			dx=kuhn_tucker_del(constr,key,j,n, H, g, x,dx,delmap)
		xi_full[idx]=dx[:n]
		OK=constr.within(x+xi_full,False)
		if OK: 
			break
		if r==k+3:
			#print('Unable to set constraints in direction calculation')
			break
	xi_full=np.zeros(m)
	xi_full[idx]=dx[:n]
	return xi_full

def kuhn_tucker_del(constr,key,j,n,H,g,x,dx,delmap,recalc=True):
	q=None
	c=constr.intervals[key]
	i=delmap[key]
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+dx[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+dx[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx=-np.linalg.solve(H,g).flatten()	
	return dx


def kuhn_tucker(c,i,j,n,H,g,x,dx,recalc=True):
	q=None
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+dx[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+dx[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx=-np.linalg.solve(H,g).flatten()	
	return dx


def kuhn_tucker2(c,i,j,n,H,g,x,dx,dx_init,recalc=True):
	if c.assco_ix is None:
		return kuhn_tucker(c,i,j,n,H,g,x,dx,recalc)
	q=None
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+dx[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+dx[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			dx=-np.linalg.solve(H,g).flatten()	
	return dx
