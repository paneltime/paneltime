#!/usr/bin/env python
# -*- coding: utf-8 -*-
import calculus
import calculus_ll as cll
import numpy as np
from scipy import stats
import constraints
import loglikelihood as logl
import time

EPS=3.0e-16 
TOLX=(4*EPS) 
STPMX=100.0 

def callback_print(percent, text, task):
	if percent is None:
		print (f"{task}: {text}")
	return True


class Computation:
	def __init__(self,panel, callback = None, id=0, mp_callback=None):
		"""callback is a function taking arguments (percent:float, text:str, task:str)"""
		self.callback = callback
		if callback == 'print':
			self.callback = callback_print
		elif callback is None:
			self.callback = (lambda percent, text, task: True)
		self.gradient=calculus.gradient(panel, self.callback)
		self.hessian=calculus.hessian(panel,self.gradient, self.callback)
		self.panel=panel
		self.constr=None
		self.mp_callback = mp_callback
		self.id = id
		self.hessian_num=None
		self.do_shocks=True
		self.input_old=None
		self.CI=0
		self.dx_norm=None
		self.dx=None
		self.H=None
		self.G=None
		self.g=None
		self.weak_mc_dict=[]
		self.mc_problems=[]
		self.H_correl_problem=False
		self.singularity_problems=False
		self.Hess_correction=1
		self.g_rec=[]
		self.start_time=time.time()
		self.minincr=0.1
		self.ll = None
		self.t = 0


	def set(self, its,increment,lmbda,rev, add_dyn_constr):
		#Assumes self.LL has been invoked
		self.its=its
		self.lmbda=lmbda
		self.has_reversed_directions=rev
		self.increment=increment
		self.constr_old=self.constr
		self.constr=constraints.constraints(self.panel,self.ll.args,its)
		self.constr.add_static_constraints(self.ll)		
		
		if add_dyn_constr:
			self.constr.add_dynamic_constraints(self)	
			
		self.CI=self.constr.CI
		if self.constr is None:
			self.H_correl_problem,self.mc_problems,self.weak_mc_dict=False, [],{}
		else:
			self.H_correl_problem=self.constr.H_correl_problem	
			self.mc_problems=self.constr.mc_problems
			self.weak_mc_dict=self.constr.weak_mc_dict
		self.singularity_problems=(len(self.mc_problems)>0) or self.H_correl_problem

	def exec(self, dx, hessin, H, its, ls, increment, force, msg=''):
		self.callback(None, ls.f, 'LL')
		p = 10
		calc_h = its/p==int(its/p)
		calc_h = False 
		
		self.calc_gradient()
		if calc_h:
			self.calc_hessian()
			
		self.set( its, increment,ls.alam,ls.rev, False)
		hessin=self.hessin_num(hessin, self.g-ls.g, dx)
		g = self.g
		return ls.x, ls.f, hessin, H, g
			
	def slave_callback(self, x, f, hessin, H, g, force):
		if (time.time()-self.t < 0.05 and (force == 0)) or (self.mp_callback is None):
			return x, f, hessin, H, g
		self.t = time.time()
		d = self.mp_callback(
			{'x':x, 'hessin':hessin, 'f':f})
		if d['f']>f or (d['reset']==self.id):
			print(f"improved by {d['f']-f} to {d['f']}")
			x, f = d['x'], d['f']
			if not d['hessin'] is None:
				hessin = d['hessin']
			if not d['H'] is None:
				H = d['H']
		return x, f, hessin, H, g
	
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
				
				
	

	def calc_hessian(self):
		d2LL_de2, d2LL_dln_de, d2LL_dln2 = cll.hessian(self.ll,self.panel)
		self.LL_hessian_tobit(self.ll, d2LL_de2, d2LL_dln_de, d2LL_dln2)
		self.H = self.hessian.get(self.ll,d2LL_de2,d2LL_dln_de,d2LL_dln2)	

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
		self.constr=constraints.constraints(self.panel, args)
		self.constr.add_static_constraints()			
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
		self.ll=ll
		return ll
	
	def calc_init_dir(self, p0):
		"""Calculates the initial computation"""
		n=len(p0)
		self.LL(p0)
		self.calc_gradient()
		self.calc_hessian()
		h = np.diag(self.H)
		h = h - (h == 0)
		self.hessin = np.diag(1/h)
		self.hessin = self.hessin*0.75 - np.identity(n)*0.25	
		return p0, self.ll.LL ,self.g, self.hessin
		
		dx=np.zeros(n)
		r=np.arange(n)
		d=0.000001
		fps=np.zeros(n)
		xs=[]
		gs=[]
		dgs=[]
		dgi=np.zeros(n)
		g0 = self.gradient.get(self.ll)
		for i in range(len(p0)):
			e = (r==i)*d
			fp = self.LL(p0+e)
			g = self.gradient.get(self.ll)
			dg = (g-g0)/d
			if dg[i]!=0:
				dx[i]  = -g0[i]/dg[i]
				dgi[i] = dg[i]
			fps[i]=fp
			xs.append(p0+e)
			gs.append(g)
			dgs.append(dg)
			#print(f"dg: {dg[i]} g:{g[i]}")
		i = np.argsort(fps)[-1]
		x = xs[i]
		fp = fps[i]
		fp = self.LL(x)
		g = gs[i]
		hessin = 1/(dgi-(dgi==0))
		hessin = np.diag(hessin)
		hessin = hessin*0.75 - np.identity(n)*0.25
		return x, fp,g, hessin

	def LL(self,x):
		if self.ll is None:
			self.init_ll(x)
			return self.ll.LL
		ll = logl.LL(x, self.panel, self.constr)
		if ll is None:
			return None
		elif ll.LL is None:
			return None
		self.ll = ll
		return ll.LL
	
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


		
def hess_inv(h, hessin):
	try:
		h_inv = np.linalg.inv(h)
	except Exception as e:
		print(e)
		return hessin
	return h_inv