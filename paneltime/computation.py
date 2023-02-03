#!/usr/bin/env python
# -*- coding: utf-8 -*-
import calculus
import calculus_ll as cll
import numpy as np
import constraints
import loglikelihood as logl
import time
import stat_dist

EPS=3.0e-16 
TOLX=(4*EPS) 
STPMX=100.0 



class Computation:
	def __init__(self,panel, gtol, tolx, callback, nummerical, diag_hess):
		"""callback is a function taking any named arguments """
		if callback is None:
			callback = lambda **kwargs: None#function that does nothing
		self.callback = callback
		self.gradient=calculus.gradient(panel, self.callback)
		self.gtol = gtol
		self.tolx = tolx
		self.hessian=calculus.hessian(panel,self.gradient, self.callback)
		self.panel=panel
		self.constr=None
		self.CI=0
		self.weak_mc_dict={}
		self.mc_problems=[]
		self.H_correl_problem=False
		self.singularity_problems=False
		self.num_hess_count = 0
		self.H, self.g, self.G = None, None, None
		self.mcollcheck = False
		self.rec =[]
		self.nummerical = nummerical
		self.quit = False
		self.avg_incr = 0
		self.ev_constr = False
		self.diag_hess = diag_hess
		self.CI_anal = 0



	def set(self, its,increment,lmbda,rev, add_dyn_constr, H, ll):
		self.its=its
		self.lmbda=lmbda
		self.has_reversed_directions=rev
		self.increment=increment
		self.constr_old=self.constr
		self.constr = constraints.Constraints(self.panel,ll.args,its)
			
		if self.constr is None:
			self.H_correl_problem,self.mc_problems,self.weak_mc_dict=False, [],{}
			self.singularity_problems=(len(self.mc_problems)>0) or self.H_correl_problem
			return
			
		if self.panel.options.constraints_engine.value:
			self.constr.add_static_constraints(self.panel, its, ll)	
			self.constr.add_dynamic_constraints(self, H, ll)	
			
		self.CI=self.constr.CI
		self.H_correl_problem=self.constr.H_correl_problem	
		self.mc_problems=self.constr.mc_problems
		self.weak_mc_dict=self.constr.weak_mc_dict
		

	def exec(self, dx, dx_norm, hessin, H, f, x, g_old, incr, rev, alam, its, ll, calc = True):
		#Thhese setting may not hold for all circumstances, and should be tested properly:
		NUM_ITER = 5
		TOTP_TOL = 0.0001
		
		g, G = self.calc_gradient(ll)
		
		pgain, totpgain = potential_gain(dx, g, H)
		self.totpgain = totpgain
		max_pgain = abs(max(pgain))
		
		a = np.ones(len(g))
		if not self.constr is None:
			a[list(self.constr.fixed.keys())] =0		
		g_norm =np.max(np.abs(g*a*x)/(abs(f)+1e-12) )
		
		self.rec.append(str(i) for i in [totpgain, max_pgain, incr, g_norm])
		CI=0
		if not self.CI is None:
			CI = self.CI

		if its >NUM_ITER and ((abs(g_norm) < self.gtol) or (abs(totpgain)<TOTP_TOL)):
			Ha = self.calc_hessian(ll)
			self.CI_anal = condition_index(Ha)
			if abs(g_norm) < self.gtol:
				return x, f, hessin, Ha, g, 1
			elif abs(totpgain)<TOTP_TOL:
				return x, f, hessin, Ha, g, 2
			
		self.avg_incr = incr + self.avg_incr*0.7
		self.ev_constr = False#its>20
		
		err = np.max(np.abs(dx)) < TOLX
		

			
		calchess = err #or ((self.CI>10000) and (self.num_hess_count>1))
		#calchess = ((abs(g_norm) <0.1 or max_pgain<10 or self.avg_incr<1) and (self.num_hess_count>10))  and False
		#print(f"{its}: {f}\tx: {x[:3]}")
			
		if calc:
			if calchess:
				print('Anal hess')
				Ha = self.calc_hessian(ll)
				self.num_hess_count = 0
				if True:
					try:
						hessin=self.hessin_num(hessin, g-g_old, dx)
						Hn = np.linalg.inv(hessin)
					except:
						Hn = H
					a = 0
					H = (Hn * a + Ha * (1-a))
				else:
					H = Ha
			else:
				self.num_hess_count +=1
				try:
					hessin=self.hessin_num(hessin, g-g_old, dx)
					Hn = np.linalg.inv(hessin)
					H = Hn
				except:
					H = self.calc_hessian(ll)

		self.H, self.g, self.G = H, g, G
			
		self.set(its, incr, alam, rev, True, H, ll)
		
	
		
		return x, f, hessin, H, g, 0



	def calc_gradient(self,ll):
		dLL_lnv, DLL_e=cll.gradient(ll,self.panel)
		self.LL_gradient_tobit(ll, DLL_e, dLL_lnv)
		g, G = self.gradient.get(ll,DLL_e,dLL_lnv,return_G=True)
		return g, G
	

	def calc_hessian(self, ll):
		d2LL_de2, d2LL_dln_de, d2LL_dln2 = cll.hessian(ll,self.panel)
		self.LL_hessian_tobit(ll, d2LL_de2, d2LL_dln_de, d2LL_dln2)
		H = self.hessian.get(ll,d2LL_de2,d2LL_dln_de,d2LL_dln2)	
		return H
	
	def LL_gradient_tobit(self,ll,DLL_e,dLL_lnv):
		g=[1,-1]
		self.f=[None,None]
		self.f_F=[None,None]
		for i in [0,1]:
			if self.panel.tobit_active[i]:
				I=self.panel.tobit_I[i]
				self.f[i]=stat_dist.norm(g[i]*ll.e_norm[I], cdf = False)
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
				
	
	def hessin_num(self, hessin, dg, xi):				#Compute difference of gradients,
		n=len(dg)
		#and difference times current matrix:
		hdg=(np.dot(hessin,dg.reshape(n,1))).flatten()
		fac=fae=sumdg=sumxi=0.0 							#Calculate dot products for the denominators. 
		fac = np.sum(dg*xi) 
		fae = np.sum(dg*hdg)
		sumdg = np.sum(dg*dg) 
		sumxi = np.sum(xi*xi) 
		if (fac < (EPS*sumdg*sumxi)**0.5):  					#Skip update if fac not sufficiently positive.
			fac=1.0/fac 
			fad=1.0/fae 
															#The vector that makes BFGS different from DFP:
			dg=fac*xi-fad*hdg   
			#The BFGS updating formula:
			hessin+=fac*xi.reshape(n,1)*xi.reshape(1,n)
			hessin-=fad*hdg.reshape(n,1)*hdg.reshape(1,n)
			hessin+=fae*dg.reshape(n,1)*dg.reshape(1,n)		
			
		return hessin

	
	def init_ll(self,args=None, ll=None):
		if args is None:
			args = ll.args.args_v
		self.constr=constraints.Constraints(self.panel, args)
		self.constr.add_static_constraints(self.panel, 0)			
		try:
			args=args.args_v
		except:
			pass#args must be a vector
		for i in self.constr.fixed:
			args[i]=self.constr.fixed[i].value
		if ll is None:
			ll=logl.LL(args, self.panel, constraints=self.constr,print_err=True)
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
		return ll
	
	def calc_init_dir(self, p0, full = False):
		"""Calculates the initial computation""" 
		ll = self.init_ll(p0)
		g, G = self.calc_gradient(ll)
		H = self.calc_hessian(ll)
		if self.diag_hess:
			d = np.diag(H)
			d = d - (np.abs(d)<1e-100)
			H = np.diag(d)
			hessin = np.diag(1/d)
		else:
			hessin = np.linalg.inv(H)
		return p0, ll, ll.LL , g, hessin, H
	
def det_managed(H):
	try:
		return np.linalg.det(H)
	except:
		return 1e+100
	
def inv_hess(hessian):
	try:
		h=-np.linalg.inv(hessian)
	except:
		return None	
	return h

def condition_index(H):
	n = len(H)
	d=np.maximum(np.abs(np.diag(H)).reshape((n,1)),1e-30)**0.5
	C = -H/(d*d.T)
	ev = np.abs(np.linalg.eigvals(C))**0.5
	if min(ev) == 0:
		return 1e-150
	return max(ev)/min(ev)	

		
def hess_inv(h, hessin):
	try:
		h_inv = np.linalg.inv(h)
	except Exception as e:
		print(e)
		return hessin
	return h_inv



def potential_gain(dx, g, H):
	"""Returns the potential gain of including each variables, given that all other variables are included and that the 
	quadratic model is correct. An alternative convercence criteria"""
	n=len(dx)
	rng=np.arange(len(dx))
	dxLL=dx*0
	dxLL_full=(sum(g*dx)+0.5*np.dot(dx.reshape((1,n)),np.dot(H,dx.reshape((n,1)))))[0,0]
	for i in range(len(dx)):
		dxi=dx*(rng!=i)
		dxLL[i]=dxLL_full-(sum(g*dxi)+0.5*np.dot(dxi.reshape((1,n)),np.dot(H,dxi.reshape((n,1)))))[0,0]
		
	return dxLL, dxLL_full



