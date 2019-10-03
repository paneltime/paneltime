#!/usr/bin/env python
# -*- coding: utf-8 -*-
import calculus
import numpy as np
import constraints as cnstr
import loglikelihood as logl



class direction:
	def __init__(self,panel):
		self.gradient=calculus.gradient(panel)
		self.hessian=calculus.hessian(panel,self.gradient)
		self.panel=panel
		self.constr=None
		self.hessian_num=None
		self.do_shocks=True
		self.input_old=None
		self.CI=1e+20
		self.I=np.diag(np.ones(panel.args.n_args))


	def get(self,ll,args,dx_norm,its,mp,dxi,numerical,no_increses):
	
		
		if its==0:
			ll=self.init_ll(args)
		if ll.LL is None:
			raise RuntimeError("Error in LL calculation: %s" %(ll.errmsg,))
		self.constr_old=self.constr
		
		self.constr=cnstr.constraints(self.panel,ll.args_v)
		cnstr.add_static_constraints(self.constr,self.panel,ll,its)			

		g,G=self.get_gradient(ll)
		hessian=self.get_hessian(ll,mp,g,G,dxi,its,dx_norm,numerical,self.CI)
		
		dx=solve(self.constr,hessian, g, ll.args_v)
		dx_norm=self.normalize(dx, ll.args_v)
		cnstr.add_dynamic_constraints(G,self.panel,ll,self.constr,dx_norm,hessian,its,self.constr_old)		
		g,G,hessian,ll=self.avoid_multicollinarity(g,G,hessian,ll)
		dx=self.remove_neg_slope(g, hessian, ll)
		dx_norm=self.normalize(dx, ll.args_v)
		

		return dx,g,G,hessian,self.constr,ll
	
	def avoid_multicollinarity(self,g,G,hessian,ll):
		limit=10000	
		if self.constr.CI>limit and (not self.input_old is None) and self.CI<limit and len(self.constr.collinears)>0:
			g1,hessian1,ll1,CI1=self.input_old#reverting to previous state if condition index too high (but with new constraints)
		
			for i in range(0,15):
				a=0.9*0.75**(i**2)
				h=a*hessian+(1-a)*hessian1
				c_index, var_prop,includemap=cnstr.decomposition(h, self.include())
				if c_index[-1]<limit:
					break
			if c_index[-1]>limit:
				a=0
			ll=logl.LL(a*ll.args_v+(1-a)*ll1.args_v,self.panel,self.constr)
			g,G=self.get_gradient(ll)
			hessian=self.get_hessian(ll,mp,g,G,dxi,its,dx_norm,numerical,self.CI)
			c_index, var_prop,includemap=cnstr.decomposition(hessian, self.include())
			self.constr.reset_collinears(ll.args_v)		
			self.CI=c_index[-1]
			if self.CI<limit:
				self.input_old=g,hessian,ll,self.constr.CI			
		else:
			self.CI=self.constr.CI
		return g,G,hessian,ll
		
	
	def include(self,all=False):
		include=np.array(self.panel.args.n_args*[True])
		if all:
			return include
		include[list(self.constr.constraints)]=False
		return include	

	def get_gradient(self,ll):
		DLL_e=-(ll.e_RE*ll.v_inv)*self.panel.included
		dLL_lnv=-0.5*(self.panel.included-(ll.e_REsq*ll.v_inv)*self.panel.included)		
		g,G=self.gradient.get(ll,DLL_e,dLL_lnv,return_G=True)	
		return g,G
		

	def get_hessian(self,ll,mp,g,G,dxi,its,dx_norm,numerical,CI):

		hessian=None
		I=self.I
		d2LL_de2=-ll.v_inv*self.panel.included
		d2LL_dln_de=ll.e_RE*ll.v_inv*self.panel.included
		d2LL_dln2=-0.5*ll.e_REsq*ll.v_inv*self.panel.included	
		if not numerical or self.hessian_num is None:
			hessian=self.hessian.get(ll,mp,d2LL_de2,d2LL_dln_de,d2LL_dln2)
			self.hessian_analytical=hessian
		else:
			hessian=self.nummerical_hessian(dxi,g)
			return hessian
		self.hessian_num=hessian
		self.g_old=g
		if (its>3) or CI<1e+10:
			return hessian
		if dx_norm is None:
			m=10
		else:
			m=max(dx_norm)**2
		hessian=(hessian+m*I*hessian)/(1+m)
		return hessian

	def nummerical_hessian(self,dxi,g):

		I=self.I
		if (self.g_old is None) or (dxi is None):
			return I

		#print("Using numerical hessian")		
		hessin_num=hessin(self.hessian_analytical)
		if hessin_num is None:
			return I
		hessin_num=nummerical_hessin(g,self.g_old,hessin_num,dxi)	
		hessian=hessin(hessin_num)
		if hessian is None:
			return I
		return hessian
	
	
	
	def init_ll(self,args):
		self.constr=cnstr.constraints(self.panel,args)
		cnstr.add_static_constraints(self.constr,self.panel,None,0)	
		ll=logl.LL(args, self.panel, constraints=self.constr)
		if ll.LL is None:
			print("""You requested stored arguments from a previous session 
			to be used as initial arguments (loadargs=True) but these failed to 
			return a valid log likelihood with the new parameters. Default inital 
			arguments will be used. """)
			ll=logl.LL(self.panel.args.args_init,self.panel,constraints=self.constr)	
		return ll
	
	def normalize(self,dx,args_v):
		dx_norm=(args_v!=0)*dx/(np.abs(args_v)+(args_v==0))
		dx_norm=(args_v<1e-2)*dx+(args_v>=1e-2)*dx_norm	
		return dx_norm	
		

	def remove_neg_slope(self,g,hessian,ll):
		include=np.ones(len(g))
		dx=solve(self.constr,hessian, g, ll.args_v)
		#return dx
		for j in range(len(dx)):
			s=dx*g*include
			if np.sum(s)<0:#negative slope
				s=np.argsort(s)
				k=s[0]
				self.constr.add(k, None, 'neg. slope')
				include[k]=False
				dx=solve(self.constr,hessian, g, ll.args_v)
			else:
				break
		return dx

def hessin(hessian):
	try:
		h=-np.linalg.inv(hessian)
	except:
		return None	
	return h

def nummerical_hessin(g,g_old,hessin,dxi):
	if dxi is None:
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


def solve(constr,H, g, x):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	if H is None:
		return None,g*0
	n=len(H)
	k=len(constr.constraints)
	H=np.concatenate((H,np.zeros((n,k))),1)
	H=np.concatenate((H,np.zeros((k,n+k))),0)
	g=np.append(g,(k)*[0])
	
	for i in range(k):
		H[n+i,n+i]=1
	j=0
	xi=np.zeros(len(g))
	for i in constr.fixed:
		kuhn_tucker(constr.fixed[i],i,j,n, H, g, x,xi, recalc=False)
		j+=1
	xi=-np.linalg.solve(H,g).flatten()	
	OK=False
	w=0
	for r in range(50):
		j2=j
		
		for i in constr.intervals:
			xi=kuhn_tucker(constr.intervals[i],i,j2,n, H, g, x,xi)
			j2+=1
		OK=constr.within(x+xi[:n],False)
		if OK: 
			break
		if r==k+3:
			#print('Unable to set constraints in direction calculation')
			break

	return xi[:n]


def kuhn_tucker(c,i,j,n,H,g,x,xi,recalc=True):
	q=None
	if not c.value is None:
		q=-(c.value-x[i])
	elif x[i]+xi[i]<c.min:
		q=-(c.min-x[i])
	elif x[i]+xi[i]>c.max:
		q=-(c.max-x[i])
	if q!=None:
		H[i,n+j]=1
		H[n+j,i]=1
		H[n+j,n+j]=0
		g[n+j]=q
		if recalc:
			xi=-np.linalg.solve(H,g).flatten()	
	return xi


