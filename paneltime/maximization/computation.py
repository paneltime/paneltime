#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..output import stat_functions as stat
from ..output import stat_dist
from .. import functions as fu
from . import constraints
from .. import likelihood as logl
import sys

from . import direction

import numpy as np


EPS=3.0e-16 
TOLX=(4*EPS) 
STPMX=100.0 



class Computation:
	def __init__(self,args, panel, gtol, tolx):
		self.gradient=logl.calculus.gradient(panel)
		self.gtol = panel.options.tolerance
		self.tolx = tolx
		self.hessian=logl.hessian(panel,self.gradient)
		self.panel=panel
		self.ci=0
		self.mc_report={}
		self.mc_report=[]
		self.H_correl_problem=False
		self.singularity_problems=False
		self.H, self.g, self.G = None, None, None
		self.mcollcheck = False
		self.rec =[]
		self.quit = False
		self.avg_incr = 0
		self.errs = []
		self.CI_anal = 2
		p, q, d, k, m = panel.pqdkm
		self.init_arma_its = 0
		self.multicoll_threshold_max = panel.options.multicoll_threshold_max
		self.set_constr(args,  panel.options.ARMA_constraint)
		





	def set(self, its,increment,lmbda,rev, H, ll, x, armaconstr):
		self.its=its
		self.lmbda=lmbda
		self.has_reversed_directions=rev
		self.increment=increment
	
		self.constr_old=self.constr
		self.constr = None
		self.constr = constraints.Constraints(self.panel,x,its,armaconstr)

		if self.constr is None:
			self.H_correl_problem,self.mc_report,self.mc_report=False, [],{}
			self.singularity_problems=(len(self.mc_report)>0) or self.H_correl_problem
			return

		if self.panel.options.constraints_engine:
			self.constr.add_static_constraints(self, its, ll)	

			self.constr.add_dynamic_constraints(self, H, ll)	

		self.ci=self.constr.ci
		if False:
			print(self.ci)
			print(x)
			print(len(self.constr.fixed))
		self.H_correl_problem=self.constr.H_correl_problem	
		self.mc_report=self.constr.mc_report
		self.mc_report=self.constr.mc_report
		


	

	def exec(self, dx_realized,hessin, H, incr, its, ls, armaconstr):
		f, x, g_old, ll = ls.f, ls.x, ls.g, ls.ll
		#These setting may not hold for all circumstances, and should be tested properly:


		g, G = self.calc_gradient(ll)
		
		hessin, H  = self.hessin_get(g, g_old, dx_realized, ll, hessin, H, its)

		self.set(its, ls.f - f, ls.alam, ls.rev,  H, ls.ll, ls.x, armaconstr)
		dx, dx_norm, H_ = direction.get(g, x, H, self.constr, f, hessin, simple=False)
		a = np.ones(len(g))
		if not self.constr is None:
			a[list(self.constr.fixed.keys())] =0		
			a[ls.applied_constraints] = 0    
		pgain, totpgain = potential_gain(dx*a, g, H)
		max_pgain = max(pgain)
		g_norm =min((np.max(np.abs(g*a*x)/(abs(f)+1e-12) ), 1e+50))
		err = np.max(np.abs(dx_realized)) < 10000*TOLX
		self.errs.append(err)
		
		if not self.panel.options.supress_output:
			print(f"its:{its}, f:{f}, gnorm: {abs(g_norm)}, totpgain: {abs(totpgain)}, max_pgain: {max(np.abs(pgain))}")
			sys.stdout.flush()

		if its<len(self.constr.constr_matrix)+2:
			conv = 0
		elif abs(g_norm*max_pgain*totpgain) < self.gtol and its>50:
			conv = 1
		elif abs(g_norm)<self.gtol:
			conv = 2
		elif its>=self.panel.options.max_iterations:
			conv = 3
		elif ((sum(self.errs[-3:])==3) and  incr<1e-15): #stalled: 3 consectutive errors and small ls.alam, or no function increase
			conv = 4
		elif (np.max(np.abs(g))>1e+50) or (np.max(np.abs(ll.e))>1e+50):
			conv = 5
		else:
			conv = 0

		if (its>100 and (incr<0.01 or ls.alam<1e-5 or sum(self.errs[-3:])==3) 
	  			and (self.multicoll_threshold_max != 1000)):
			self.multicoll_threshold_max = 1000
		elif its>100:
			self.multicoll_threshold_max = self.panel.options.multicoll_threshold_max
		
		return x, f, hessin, H, G, g, conv, g_norm, dx
	


	def fixed_constr_change(self):
		new = self.constr.fixed.keys()
		old = self.constr_old.fixed.keys()
		for i in new:
			if not i in old:
				return True
		for i in old:
			if not i in new:
				return True

		return False

	
	def set_constr(self, args, armaconstr):
		self.constr_old = None
		self.constr = constraints.Constraints(self.panel, args, 0, armaconstr)
		self.constr.add_static_constraints(self, 0)	    


		
	def calc_gradient(self,ll):
		dLL_lnv, DLL_e=ll.llfunc.gradient()
		self.LL_gradient_tobit(ll, DLL_e, dLL_lnv)
		g, G = self.gradient.get(ll,DLL_e,dLL_lnv,return_G=True)
		return g, G


	def calc_hessian(self, ll):
		d2LL_de2, d2LL_dln_de, d2LL_dln2 = ll.llfunc.hessian()
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

				DLL_e[I]=g[i]*self.f_F[i]*self.llfunc.v_inv05[I]
				dLL_lnv[I]=-0.5*DLL_e[I]*ll.e_RE[I]
				a=0


	def LL_hessian_tobit(self,ll,d2LL_de2,d2LL_dln_de,d2LL_dln2):
		g=[1,-1]
		if sum(self.panel.tobit_active)==0:
			return
		self.f=[None,None]
		e1s1=ll.e_norm
		e2s2=ll.e2*ll.llfunc.v_inv
		e3s3=e2s2*e1s1
		e1s2=e1s1*self.llfunc.v_inv05
		e1s3=e1s1*ll.llfunc.v_inv
		e2s3=e2s2*self.llfunc.v_inv05
		f_F=self.f_F
		for i in [0,1]:
			if self.panel.tobit_active[i]:
				I=self.panel.tobit_I[i]
				f_F2=self.f_F[i]**2
				d2LL_de2[I]=      -(g[i]*f_F[i]*e1s3[I] + f_F2*ll.llfunc.v_inv[I])
				d2LL_dln_de[I] =   0.5*(f_F2*e1s2[I]  +  g[i]*f_F[i]*(e2s3[I]-self.llfunc.v_inv05[I]))
				d2LL_dln2[I] =     0.25*(f_F2*e2s2[I]  +  g[i]*f_F[i]*(e1s1[I]-e3s3[I]))


	def hessin_num(self, hessin, dg, xi):				#Compute difference of gradients,
		n=len(dg)
		#and difference times current matrix:
		hdg=(np.dot(hessin,dg.reshape(n,1))).flatten()
		#Calculate dot products for the denominators. 
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

	def hessin_get(self, g, g_old, dx_realized, ll, hessin_orig, H_orig, its):
		H = self.calc_hessian(ll)
		Hn, hessin = None, None

		try:
			hessin=self.hessin_num(hessin_orig, g-g_old, dx_realized)
			Hn = np.linalg.inv(hessin)
		except:
			pass
		if hessin is None or Hn is None:
			if np.linalg.det(H)<0:
				hessin = np.linalg.inv(H)
				Hn = H
			else:
				if np.sum((g-g_old)*dx_realized)==0:
					hessin = -np.identity(len(hessin_orig))
				else:
					hessin=self.hessin_num(np.identity(len(hessin_orig)), g-g_old, dx_realized)
					if abs(np.linalg.det(hessin))<1e-100:
						hessin = - np.identity(len(hessin_orig))
				Hn = np.linalg.inv(hessin)
		if H is None:
			H = Hn

		H, hessin = self.mixedhess(H, Hn, hessin, its)

		return hessin, H
	

	def mixedhess(self, H, Hn, hessin, its):

		if self.panel.options.use_analytical==0 and its>5:
			try:
				H_det = abs(fu.try_warn(np.linalg.det, (H,)))
				Hn_det = abs(fu.try_warn(np.linalg.det, (Hn,)))
				if H_det*Hn_det>0:
					a = (1/H_det)/((1/H_det)+(1/Hn_det))
					a = max((a,0.5))
					H = a*H + (1-a)*hessin
			except:
				H = 0.5* H + 0.5*Hn
		elif self.panel.options.use_analytical==1:
			H = 0.5* H + 0.5*Hn
		try:
			hessin = np.linalg.inv(H)
		except:
			pass

		return H, hessin



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
	dxLL_full=(sum(g*dx)+0.5*np.dot(dx.reshape((1,n)),
																	np.dot(H,dx.reshape((n,1)))
																	))[0,0]
	for i in range(len(dx)):
		dxi=dx*(rng!=i)
		dxLL[i]=dxLL_full-(sum(g*dxi)+0.5*np.dot(dxi.reshape((1,n)),np.dot(H,dxi.reshape((n,1)))))[0,0]

	dxLL = np.minimum(np.abs(dxLL), 1e+50)
	dxLL_full = min((np.abs(dxLL_full), 1e+50))
	return dxLL, dxLL_full



