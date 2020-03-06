#!/usr/bin/env python
# -*- coding: utf-8 -*-

import calculus_functions as cf
import numpy as np
import time
import debug
import os
import multi_core
import loglikelihood as logl

class gradient:
	
	def __init__(self,panel,progress_bar):
		self.panel=panel
		self.progress_bar=progress_bar
		
	def arima_grad(self,k,x,sign=1,pre=None):
		if k==0:
			return None

		(N,T,m)=x.shape
		#L is "T x T" 
		#x is "N x T x 1"  
		#creates a  "k x T x N x 1": 
		x=np.array([cf.roll(x,i+1,1) for i in range(k)])
		#reshapes to  "N x T x k": 
		x=np.swapaxes(x,0,3).reshape((N,T,k))
		if not pre is None:
			x=cf.dot(pre,x)
		if sign<0:
			x=x*sign
		return x*self.panel.a

	def garch_arima_grad(self,ll,d,dRE,varname):
		panel=self.panel
		groupeffect=0
		groupeffect, dvRE_dx=None, None
		d_input=0
		if self.panel.N>1 and panel.settings.group_fixed_random_eff.value>0 and not dRE is None:
			d_eRE_sq=2*ll.e_RE*dRE
			dmeane2=panel.mean(d_eRE_sq,(0,1))
			d_input=(d_eRE_sq-dmeane2)*panel.a
			dvRE_dx=dmeane2*panel.a-ll.re_obj_i_v.dRE(d_input,ll.varRE_input,varname)-ll.re_obj_t_v.dRE(d_input,ll.varRE_input,varname)
			groupeffect=ll.dlnvRE*dvRE_dx*panel.a
			
		if self.panel.pqdkm[4]>0 and not d is None:
			((N,T,k))=d.shape
			x=cf.prod((ll.h_e_val,d))
			dlnv_sigma_G=cf.dot(ll.GAR_1MA,x)


			dlnv_e=cf.add((dlnv_sigma_G,groupeffect),True)
			return dlnv_e,dlnv_sigma_G,dvRE_dx,d_input
		else:
			return groupeffect,None,dvRE_dx,d_input

	def get(self,ll,DLL_e=None,dLL_lnv=None,return_G=False):
		if not self.progress_bar(0.05,'Calculating the gradient'):return
		(self.DLL_e, self.dLL_lnv)=(DLL_e, dLL_lnv)
		panel=self.panel
		re_obj_i,re_obj_t=ll.re_obj_i,ll.re_obj_t
		u,e,h_e_val,lnv_ARMA,h_val,v=ll.u,ll.e,ll.h_e_val,ll.lnv_ARMA,ll.h_val,ll.v
		p,q,d,k,m=panel.pqdkm
		nW=panel.nW
		if DLL_e is None:
			DLL_e=-(ll.e_RE*ll.v_inv)*self.panel.included
			dLL_lnv=-0.5*(self.panel.included-(ll.e_REsq*ll.v_inv)*self.panel.included)	
			dLL_lnv*=ll.dlnv_pos
		#ARIMA:
		de_rho=self.arima_grad(p,u,-1,ll.AMA_1)
		de_lambda=self.arima_grad(q,e,-1,ll.AMA_1)
		de_beta=-cf.dot(ll.AMA_1AR,panel.X)*panel.a
		
		(self.de_rho,self.de_lambda,self.de_beta)=(de_rho,de_lambda,de_beta)
		
		self.de_rho_RE       =    cf.add((de_rho,     re_obj_i.dRE(de_rho, ll.e,'rho'), 		re_obj_t.dRE(de_rho,ll.e,'rho')), True)
		self.de_lambda_RE    =    cf.add((de_lambda,  re_obj_i.dRE(de_lambda, ll.e,'lambda'),	re_obj_t.dRE(de_lambda,ll.e,'lambda')), True)
		self.de_beta_RE      =    cf.add((de_beta,    re_obj_i.dRE(de_beta, ll.e,'beta'), 		re_obj_t.dRE(de_beta,ll.e,'beta')), True)		

		dlnv_sigma_rho,		dlnv_sigma_rho_G,		dvRE_rho	, d_rho_input		=	self.garch_arima_grad(ll,	de_rho,		self.de_rho_RE,		'rho')
		dlnv_sigma_lambda, 	dlnv_sigma_lambda_G,	dvRE_lambda	, d_lambda_input	=	self.garch_arima_grad(ll,	de_lambda,	self.de_lambda_RE,	'lambda')
		dlnv_sigma_beta,	dlnv_sigma_beta_G,		dvRE_beta	, d_beta_input		=	self.garch_arima_grad(ll,	de_beta,	self.de_beta_RE,	'beta')

		(self.dlnv_sigma_rho,self.dlnv_sigma_lambda,self.dlnv_sigma_beta)=(dlnv_sigma_rho,dlnv_sigma_lambda,dlnv_sigma_beta)
		(self.dlnv_sigma_rho_G,self.dlnv_sigma_lambda_G,self.dlnv_sigma_beta_G)=(dlnv_sigma_rho_G,dlnv_sigma_lambda_G,dlnv_sigma_beta_G)
		(self.dvRE_rho,self.dvRE_lambda,self.dvRE_beta)=(dvRE_rho,dvRE_lambda,dvRE_beta)
		(self.d_rho_input,self.d_lambda_input,self.d_beta_input)=(d_rho_input,d_lambda_input,d_beta_input)

		#GARCH:
		(dlnv_gamma, dlnv_psi, dlnv_mu, dlnv_z_G, dlnv_z)=(None,None,None,None,None)
		if panel.N>1:
			dlnv_mu=cf.prod((ll.dlnvRE_mu,panel.included))
		else:
			dlnv_mu=None	
			
		if m>0:
			dlnv_gamma=self.arima_grad(k,lnv_ARMA,1,ll.GAR_1)
			dlnv_psi=self.arima_grad(m,h_val,1,ll.GAR_1)
			if not ll.h_z_val is None:
				dlnv_z_G=cf.dot(ll.GAR_1MA,ll.h_z_val)
				(N,T,k)=dlnv_z_G.shape

			dlnv_z=dlnv_z_G


		(self.dlnv_gamma, self.dlnv_psi,self.dlnv_mu,self.dlnv_z_G,self.dlnv_z)=(dlnv_gamma, dlnv_psi, dlnv_mu, dlnv_z_G, dlnv_z)

		#LL

		#final derivatives:
		dLL_beta=cf.add((cf.prod((dlnv_sigma_beta,dLL_lnv)),cf.prod((self.de_beta_RE,DLL_e))),True)
		dLL_rho=cf.add((cf.prod((dlnv_sigma_rho,dLL_lnv)),cf.prod((self.de_rho_RE,DLL_e))),True)
		dLL_lambda=cf.add((cf.prod((dlnv_sigma_lambda,dLL_lnv)),cf.prod((self.de_lambda_RE,DLL_e))),True)
		dLL_gamma=cf.prod((dlnv_gamma,dLL_lnv))
		dLL_psi=cf.prod((dlnv_psi,dLL_lnv))
		dLL_omega=cf.prod((panel.W_a,dLL_lnv))
		dLL_mu=cf.prod((self.dlnv_mu,dLL_lnv))
		dLL_z=cf.prod((self.dlnv_z,dLL_lnv))

		G=cf.concat_marray((dLL_beta,dLL_rho,dLL_lambda,dLL_gamma,dLL_psi,dLL_omega,dLL_mu,dLL_z))
		g=np.sum(G,(0,1))
		#For debugging:
		#print (g)
		#gn=debug.grad_debug(ll,panel,0.0000001)#debugging
		#if np.sum((g-gn)**2)>10000000:
		#	a=0
		#print(gn)
		#a=debug.grad_debug_detail(ll, panel, 0.00000001, 'LL', 'beta',0)
		#dLLeREn,deREn=debug.LL_calc_custom(ll, panel, 0.0000001)
		if not self.progress_bar(0.08,'Calculating the hessian'):return
		if return_G:
			return  g,G
		else:
			return g


class hessian:
	def __init__(self,panel,g,progress_bar):
		self.panel=panel
		self.its=0
		self.sec_deriv=self.set_mp_strings()
		self.g=g
		self.progress_bar=progress_bar
		
	
	def get(self,ll,mp,d2LL_de2,d2LL_dln_de,d2LL_dln2):	
		if mp is None:
			return self.hessian(ll,d2LL_de2,d2LL_dln_de,d2LL_dln2)
		else:
			return self.hessian_mp(ll,mp,d2LL_de2,d2LL_dln_de,d2LL_dln2)

	def hessian(self,ll,d2LL_de2,d2LL_dln_de,d2LL_dln2):
		panel=self.panel
		tic=time.perf_counter()
		g=self.g
		p,q,d,k,m=panel.pqdkm
		GARM=cf.ARMA_product(ll.GAR_1,m)
		GARK=cf.ARMA_product(ll.GAR_1,k)

		d2lnv_gamma2		=   cf.prod((2, 
		                        cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_gamma,						g.dLL_lnv,  transpose=True)))
		d2lnv_gamma_psi		=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_psi,							g.dLL_lnv)

		d2lnv_gamma_rho		=	cf.dd_func_lags(panel,ll,GARK,	g.dlnv_sigma_rho_G,						g.dLL_lnv)
		d2lnv_gamma_lambda	=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_sigma_lambda_G,					g.dLL_lnv)
		d2lnv_gamma_beta	=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_sigma_beta_G,					g.dLL_lnv)
		d2lnv_gamma_z		=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_z_G,							g.dLL_lnv)

		d2lnv_psi_rho		=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_rho)),		g.dLL_lnv)
		d2lnv_psi_lambda	=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_lambda)),	g.dLL_lnv)
		d2lnv_psi_beta		=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_beta)),	g.dLL_lnv)
		d2lnv_psi_z			=	cf.dd_func_lags(panel,ll,GARM, 	ll.h_z_val,								g.dLL_lnv)

		AMAq=-cf.ARMA_product(ll.AMA_1,q)
		d2lnv_lambda2,		d2e_lambda2		=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'lambda', transpose=True)
		d2lnv_lambda_rho,	d2e_lambda_rho	=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'rho' )
		d2lnv_lambda_beta,	d2e_lambda_beta	=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'beta')

		AMAp=-cf.ARMA_product(ll.AMA_1,p)
		d2lnv_rho_beta,		d2e_rho_beta	=	cf.dd_func_lags_mult(panel,ll,g,AMAp,	'rho',		'beta', u_gradient=True)
		
		d2lnv_mu_rho,d2lnv_mu_lambda,d2lnv_mu_beta,d2lnv_mu_z,mu=None,None,None,None,None
		if panel.N>1:
			d2lnv_mu_rho			=	cf.sumNT(cf.prod((ll.ddlnvRE_mu_vRE, 	g.dvRE_rho,  	 	g.dLL_lnv)))
			d2lnv_mu_lambda			=	cf.sumNT(cf.prod((ll.ddlnvRE_mu_vRE, 	g.dvRE_lambda,  	g.dLL_lnv)))
			d2lnv_mu_beta			=	cf.sumNT(cf.prod((ll.ddlnvRE_mu_vRE, 	g.dvRE_beta,  	 	g.dLL_lnv)))
			d2lnv_mu_z=None
			d2lnv_mu2=0

	
		d2lnv_z2				=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, ll.h_2z_val,						g.dLL_lnv) 
		d2lnv_z_rho				=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, cf.prod((ll.h_ez_val,g.de_rho)),	g.dLL_lnv) 
		d2lnv_z_lambda			=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, cf.prod((ll.h_ez_val,g.de_lambda)),g.dLL_lnv) 
		d2lnv_z_beta			=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, cf.prod((ll.h_ez_val,g.de_beta)),	g.dLL_lnv) 
		
		d2lnv_rho2,	d2e_rho2	=	cf.dd_func_lags_mult(panel,ll,g,	None,	'rho',		'rho' )
		d2lnv_beta2,d2e_beta2	=	cf.dd_func_lags_mult(panel,ll,g,	None,	'beta',		'beta')

		(de_rho_RE,de_lambda_RE,de_beta_RE)=(g.de_rho_RE,g.de_lambda_RE,g.de_beta_RE)
		(dlnv_sigma_rho,dlnv_sigma_lambda,dlnv_sigma_beta)=(g.dlnv_sigma_rho,g.dlnv_sigma_lambda,g.dlnv_sigma_beta)
		(dlnv_gamma,dlnv_psi)=(g.dlnv_gamma,g.dlnv_psi)
		(dlnv_mu,dlnv_z)=(g.dlnv_mu, g.dlnv_z)		

		#Final:
		D2LL_beta2			=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_beta_RE,		dlnv_sigma_beta, 	dlnv_sigma_beta,	d2e_beta2, 					d2lnv_beta2)
		D2LL_beta_rho		=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_rho_RE,		dlnv_sigma_beta, 	dlnv_sigma_rho,		T(d2e_rho_beta), 		T(d2lnv_rho_beta))
		D2LL_beta_lambda	=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_lambda_RE,	dlnv_sigma_beta, 	dlnv_sigma_lambda,	T(d2e_lambda_beta), 	T(d2lnv_lambda_beta))
		D2LL_beta_gamma		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma_beta))
		D2LL_beta_psi		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	g.dlnv_psi,		None, 					T(d2lnv_psi_beta))
		D2LL_beta_omega		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	panel.W_a,		None, 					None)
		D2LL_beta_mu		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	dlnv_mu,		None, 					d2lnv_mu_beta)
		D2LL_beta_z			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	dlnv_z,			None, 					T(d2lnv_z_beta))
		
		D2LL_rho2			=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_rho_RE,		dlnv_sigma_rho, 	dlnv_sigma_rho,		d2e_rho2, 					d2lnv_rho2)
		D2LL_rho_lambda		=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_lambda_RE,	dlnv_sigma_rho, 	dlnv_sigma_lambda,	T(d2e_lambda_rho), 		T(d2lnv_lambda_rho))
		D2LL_rho_gamma		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma_rho))	
		D2LL_rho_psi		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	g.dlnv_psi,		None, 					T(d2lnv_psi_rho))
		D2LL_rho_omega		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	panel.W_a,		None, 					None)
		D2LL_rho_mu			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	dlnv_mu,		None, 					T(d2lnv_mu_rho))
		D2LL_rho_z			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	dlnv_z,			None, 					T(d2lnv_z_rho))
		
		D2LL_lambda2		=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	de_lambda_RE,	dlnv_sigma_lambda, 	dlnv_sigma_lambda,	T(d2e_lambda2), 		T(d2lnv_lambda2))
		D2LL_lambda_gamma	=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma_lambda))
		D2LL_lambda_psi		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	g.dlnv_psi,		None, 					T(d2lnv_psi_lambda))
		D2LL_lambda_omega	=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	panel.W_a,		None, 					None)
		D2LL_lambda_mu		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	dlnv_mu,		None, 					T(d2lnv_mu_lambda))
		D2LL_lambda_z		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	dlnv_z,			None, 					T(d2lnv_z_lambda))
		
		D2LL_gamma2			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma2))
		D2LL_gamma_psi		=	cf.dd_func(None,		None,			d2LL_dln2,	None,			None,			g.dlnv_gamma, 	g.dlnv_psi,		None, 					d2lnv_gamma_psi)
		D2LL_gamma_omega	=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	panel.W_a,		None, 					None)
		D2LL_gamma_mu		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	dlnv_mu,		None, 					None)
		D2LL_gamma_z		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	dlnv_z,			None, 					d2lnv_gamma_z)
		
		D2LL_psi2			=	cf.dd_func(None,		None,			d2LL_dln2,	None,			None,			g.dlnv_psi, 		g.dlnv_psi,		None, 					None)
		D2LL_psi_omega		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_psi, 		panel.W_a,		None, 					None)
		D2LL_psi_mu			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_psi, 		dlnv_mu,		None, 					None)
		D2LL_psi_z			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_psi, 		dlnv_z,			None, 					d2lnv_psi_z)
		
		D2LL_omega2			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		panel.W_a,		None, 					None)
		D2LL_omega_mu		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_mu,		None, 					None)
		D2LL_omega_z		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_z,			None, 					None)
		
		D2LL_mu2			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_mu,		None, 					None)
		D2LL_mu_z			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_z,			None, 					d2lnv_mu_z)
		
		D2LL_z2				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_z, 		dlnv_z,			None, 					d2lnv_z2)

		H= [[D2LL_beta2,			D2LL_beta_rho,		D2LL_beta_lambda,		D2LL_beta_gamma,	D2LL_beta_psi,		D2LL_beta_omega,	D2LL_beta_mu,	D2LL_beta_z		],
	        [T(D2LL_beta_rho),		D2LL_rho2,			D2LL_rho_lambda,		D2LL_rho_gamma,		D2LL_rho_psi,		D2LL_rho_omega,		D2LL_rho_mu,	D2LL_rho_z			],
	        [T(D2LL_beta_lambda),	T(D2LL_rho_lambda),	D2LL_lambda2,			D2LL_lambda_gamma,	D2LL_lambda_psi,	D2LL_lambda_omega,	D2LL_lambda_mu,	D2LL_lambda_z		],
	        [T(D2LL_beta_gamma),	T(D2LL_rho_gamma),	T(D2LL_lambda_gamma),	D2LL_gamma2,		D2LL_gamma_psi,		D2LL_gamma_omega, 	D2LL_gamma_mu,	D2LL_gamma_z		],
	        [T(D2LL_beta_psi),		T(D2LL_rho_psi),	T(D2LL_lambda_psi),		T(D2LL_gamma_psi),	D2LL_psi2,			D2LL_psi_omega, 	D2LL_psi_mu,	D2LL_psi_z			],
	        [T(D2LL_beta_omega),	T(D2LL_rho_omega),	T(D2LL_lambda_omega),	T(D2LL_gamma_omega),T(D2LL_psi_omega),	D2LL_omega2, 		D2LL_omega_mu,	D2LL_omega_z		], 
	        [T(D2LL_beta_mu),		T(D2LL_rho_mu),		T(D2LL_lambda_mu),		T(D2LL_gamma_mu),	T(D2LL_psi_mu),		T(D2LL_omega_mu), 	D2LL_mu2,		D2LL_mu_z			],
	        [T(D2LL_beta_z),		T(D2LL_rho_z),		T(D2LL_lambda_z),		T(D2LL_gamma_z),	T(D2LL_psi_z),		T(D2LL_omega_z), 	D2LL_mu_z,		D2LL_z2				]]

		H=cf.concat_matrix(H)
		#for debugging:
		#Hn=debug.hess_debug(ll,panel,g,0.00000001)#debugging
		#v=debug.hess_debug_detail(ll,panel,0.0000001,'grp','beta','beta',0,0)
		#print (time.perf_counter()-tic)
		self.its+=1
		if np.any(np.isnan(H)):
			return None
		#print(H[0]/1e+11)
		return H
	


	def second_derivatives_mp(self,ll,mp,d2LL_de2,d2LL_dln_de,d2LL_dln2):
		panel=self.panel
		g=self.g
		
		NT,k=self.panel.NT,self.panel.args.n_args
		use_mp= (NT*(k**0.5)>200000 and os.cpu_count()>=2) 
		d={'ll':ll_light(ll),'g':g_obj_light(g)}
		#if use_mp:
		mp.send_dict_by_file(d)	
		progress_bar=[self.progress_bar,0.1,0.9,'Calculating the hessian']
		t_1=time.perf_counter()
		d,t=mp.execute(self.evalstr,True,progress_bar=progress_bar)
		d['d2LL_de2']=d2LL_de2
		d['d2LL_dln_de']=d2LL_dln_de
		d['d2LL_dln2']=d2LL_dln2
		(d['de_rho_RE'],d['de_lambda_RE'],d['de_beta_RE'])=(g.de_rho_RE,g.de_lambda_RE,g.de_beta_RE)
		(d['dlnv_sigma_rho'],d['dlnv_sigma_lambda'],d['dlnv_sigma_beta'])=(g.dlnv_sigma_rho,g.dlnv_sigma_lambda,g.dlnv_sigma_beta)
		(d['dlnv_gamma'],d['dlnv_psi'])=(g.dlnv_gamma,g.dlnv_psi)

		(d['dlnv_mu'], d['dlnv_z'])=(g.dlnv_mu, g.dlnv_z)		

		return d
	
	def set_mp_strings(self):
		panel=self.panel
		#these are all "k x T x T" matrices:
		evalstr=[]		
		#strings are evaluated for the code to be compatible with multi core proccessing
		p,q,d,k,m=panel.pqdkm
		evalstr.append(f"""	                        
GARM=cf.ARMA_product(ll.GAR_1,{m})
GARK=cf.ARMA_product(ll.GAR_1,{k})

d2lnv_gamma2		=   cf.prod((2, 
		                cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_gamma,						g.dLL_lnv,  transpose=True)))
d2lnv_gamma_psi		=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_psi,							g.dLL_lnv)

d2lnv_gamma_rho		=	cf.dd_func_lags(panel,ll,GARK,	g.dlnv_sigma_rho_G,						g.dLL_lnv)
d2lnv_gamma_lambda	=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_sigma_lambda_G,					g.dLL_lnv)
d2lnv_gamma_beta	=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_sigma_beta_G,					g.dLL_lnv)
d2lnv_gamma_z		=	cf.dd_func_lags(panel,ll,GARK, 	g.dlnv_z_G,							g.dLL_lnv)

d2lnv_psi_rho		=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_rho)),		g.dLL_lnv)
d2lnv_psi_lambda	=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_lambda)),	g.dLL_lnv)
d2lnv_psi_beta		=	cf.dd_func_lags(panel,ll,GARM, 	cf.prod((ll.h_e_val,g.de_beta)),	g.dLL_lnv)
d2lnv_psi_z			=	cf.dd_func_lags(panel,ll,GARM, 	ll.h_z_val,								g.dLL_lnv)
""")
		#ARCH:
		evalstr.append(f"""
AMAq=-cf.ARMA_product(ll.AMA_1,{q})
d2lnv_lambda2,		d2e_lambda2		=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'lambda', transpose=True)
""")
		
		evalstr.append(f"""
AMAq=-cf.ARMA_product(ll.AMA_1,{q})
d2lnv_lambda_rho,	d2e_lambda_rho	=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'rho' )
""")
		
		evalstr.append(f"""
AMAq=-cf.ARMA_product(ll.AMA_1,{q})
d2lnv_lambda_beta,	d2e_lambda_beta	=	cf.dd_func_lags_mult(panel,ll,g,AMAq,	'lambda',	'beta') """)	
		
		evalstr.append(f"""		
AMAp=-cf.ARMA_product(ll.AMA_1,{p})
d2lnv_rho_beta,		d2e_rho_beta	=	cf.dd_func_lags_mult(panel,ll,g,AMAp,	'rho',		'beta', u_gradient=True)

d2lnv_mu_rho,d2lnv_mu_lambda,d2lnv_mu_beta,d2lnv_mu_z,mu=None,None,None,None,None
if panel.N>1:
	d2lnv_mu_rho			=	cf.sumNT(cf.prod((ll.ddlnvRE_mu_vRE, 	g.dvRE_rho,  	 	g.dLL_lnv)))
	d2lnv_mu_lambda			=	cf.sumNT(cf.prod((ll.ddlnvRE_mu_vRE, 	g.dvRE_lambda,  	g.dLL_lnv)))
	d2lnv_mu_beta			=	cf.sumNT(cf.prod((ll.ddlnvRE_mu_vRE, 	g.dvRE_beta,  	 	g.dLL_lnv)))
	d2lnv_mu_z=None
	d2lnv_mu2=0

d2lnv_z2				=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, ll.h_2z_val,						g.dLL_lnv) 
d2lnv_z_rho				=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, cf.prod((ll.h_ez_val,g.de_rho)),	g.dLL_lnv) 
d2lnv_z_lambda			=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, cf.prod((ll.h_ez_val,g.de_lambda)),g.dLL_lnv) 
d2lnv_z_beta			=	cf.dd_func_lags(panel,ll,ll.GAR_1MA, cf.prod((ll.h_ez_val,g.de_beta)),	g.dLL_lnv) 
""")
		
		evalstr.append("d2lnv_rho2,	d2e_rho2	=	cf.dd_func_lags_mult(panel,ll,g,	None,	'rho',		'rho' )")
		
		evalstr.append("d2lnv_beta2,d2e_beta2	=	cf.dd_func_lags_mult(panel,ll,g,	None,	'beta',		'beta')")
	
		self.evalstr=evalstr



	def hessian_mp(self,ll,mp,d2LL_de2,d2LL_dln_de,d2LL_dln2):
		panel=self.panel
		tic=time.perf_counter()
		#return debug.hessian_debug(self,args):
		d=self.second_derivatives_mp(ll,mp,d2LL_de2,d2LL_dln_de,d2LL_dln2)
		#Final: we need to pass the last code as string, since we want to use the dictionary d in the evaluation
		evalstr="""
D2LL_beta2			=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_beta_RE,		dlnv_sigma_beta, 	dlnv_sigma_beta,	d2e_beta2, 					d2lnv_beta2)
D2LL_beta_rho		=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_rho_RE,		dlnv_sigma_beta, 	dlnv_sigma_rho,		T(d2e_rho_beta), 		T(d2lnv_rho_beta))
D2LL_beta_lambda	=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_lambda_RE,	dlnv_sigma_beta, 	dlnv_sigma_lambda,	T(d2e_lambda_beta), 	T(d2lnv_lambda_beta))
D2LL_beta_gamma		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma_beta))
D2LL_beta_psi		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	g.dlnv_psi,		None, 					T(d2lnv_psi_beta))
D2LL_beta_omega		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	panel.W_a,		None, 					None)
D2LL_beta_mu		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	dlnv_mu,		None, 					d2lnv_mu_beta)
D2LL_beta_z			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_sigma_beta, 	dlnv_z,			None, 					T(d2lnv_z_beta))

D2LL_rho2			=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_rho_RE,		dlnv_sigma_rho, 	dlnv_sigma_rho,		d2e_rho2, 					d2lnv_rho2)
D2LL_rho_lambda		=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_lambda_RE,	dlnv_sigma_rho, 	dlnv_sigma_lambda,	T(d2e_lambda_rho), 		T(d2lnv_lambda_rho))
D2LL_rho_gamma		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma_rho))	
D2LL_rho_psi		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	g.dlnv_psi,		None, 					T(d2lnv_psi_rho))
D2LL_rho_omega		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	panel.W_a,		None, 					None)
D2LL_rho_mu			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	dlnv_mu,		None, 					T(d2lnv_mu_rho))
D2LL_rho_z			=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_sigma_rho, 	dlnv_z,			None, 					T(d2lnv_z_rho))

D2LL_lambda2		=	cf.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	de_lambda_RE,	dlnv_sigma_lambda, 	dlnv_sigma_lambda,	T(d2e_lambda2), 		T(d2lnv_lambda2))
D2LL_lambda_gamma	=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma_lambda))
D2LL_lambda_psi		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	g.dlnv_psi,		None, 					T(d2lnv_psi_lambda))
D2LL_lambda_omega	=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	panel.W_a,		None, 					None)
D2LL_lambda_mu		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	dlnv_mu,		None, 					T(d2lnv_mu_lambda))
D2LL_lambda_z		=	cf.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_sigma_lambda, 	dlnv_z,			None, 					T(d2lnv_z_lambda))

D2LL_gamma2			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	g.dlnv_gamma,		None, 					T(d2lnv_gamma2))
D2LL_gamma_psi		=	cf.dd_func(None,		None,			d2LL_dln2,	None,			None,			g.dlnv_gamma, 	g.dlnv_psi,		None, 					d2lnv_gamma_psi)
D2LL_gamma_omega	=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	panel.W_a,		None, 					None)
D2LL_gamma_mu		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	dlnv_mu,		None, 					None)
D2LL_gamma_z		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_gamma, 	dlnv_z,			None, 					d2lnv_gamma_z)

D2LL_psi2			=	cf.dd_func(None,		None,			d2LL_dln2,	None,			None,			g.dlnv_psi, 		g.dlnv_psi,		None, 					None)
D2LL_psi_omega		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_psi, 		panel.W_a,		None, 					None)
D2LL_psi_mu			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_psi, 		dlnv_mu,		None, 					None)
D2LL_psi_z			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			g.dlnv_psi, 		dlnv_z,			None, 					d2lnv_psi_z)

D2LL_omega2			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		panel.W_a,		None, 					None)
D2LL_omega_mu		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_mu,		None, 					None)
D2LL_omega_z		=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_z,			None, 					None)

D2LL_mu2			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_mu,		None, 					None)
D2LL_mu_z			=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_z,			None, 					d2lnv_mu_z)

D2LL_z2				=	cf.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_z, 		dlnv_z,			None, 					d2lnv_z2)

		"""
		exec(evalstr,None,d)


		evalstr="""
H= [[D2LL_beta2,			D2LL_beta_rho,		D2LL_beta_lambda,		D2LL_beta_gamma,	D2LL_beta_psi,		D2LL_beta_omega,	D2LL_beta_mu,	D2LL_beta_z		],
	[T(D2LL_beta_rho),		D2LL_rho2,			D2LL_rho_lambda,		D2LL_rho_gamma,		D2LL_rho_psi,		D2LL_rho_omega,		D2LL_rho_mu,	D2LL_rho_z			],
	[T(D2LL_beta_lambda),	T(D2LL_rho_lambda),	D2LL_lambda2,			D2LL_lambda_gamma,	D2LL_lambda_psi,	D2LL_lambda_omega,	D2LL_lambda_mu,	D2LL_lambda_z		],
	[T(D2LL_beta_gamma),	T(D2LL_rho_gamma),	T(D2LL_lambda_gamma),	D2LL_gamma2,		D2LL_gamma_psi,		D2LL_gamma_omega, 	D2LL_gamma_mu,	D2LL_gamma_z		],
	[T(D2LL_beta_psi),		T(D2LL_rho_psi),	T(D2LL_lambda_psi),		T(D2LL_gamma_psi),	D2LL_psi2,			D2LL_psi_omega, 	D2LL_psi_mu,	D2LL_psi_z			],
	[T(D2LL_beta_omega),	T(D2LL_rho_omega),	T(D2LL_lambda_omega),	T(D2LL_gamma_omega),T(D2LL_psi_omega),	D2LL_omega2, 		D2LL_omega_mu,	D2LL_omega_z		], 
	[T(D2LL_beta_mu),		T(D2LL_rho_mu),		T(D2LL_lambda_mu),		T(D2LL_gamma_mu),	T(D2LL_psi_mu),		T(D2LL_omega_mu), 	D2LL_mu2,		D2LL_mu_z			],
	[T(D2LL_beta_z),		T(D2LL_rho_z),		T(D2LL_lambda_z),		T(D2LL_gamma_z),	T(D2LL_psi_z),		T(D2LL_omega_z), 	D2LL_mu_z,		D2LL_z2				]]
		"""
		exec(evalstr,None,d)
		H=d['H']
		H=cf.concat_matrix(H)
		if np.any(np.isnan(H)):
			return None
		a=False
		if a==True:
			Hn=debug.hess_debug(ll,panel,self.g,0.000000001)#debugging
		#for debugging:
		#Hn=debug.hess_debug(ll,panel,self.g,0.000000001)#debugging
		#H_debug=hessian(self, ll)
		#debug.LL_debug_detail(self,ll,0.0000001)
		#print (time.perf_counter()-tic)
		self.its+=1
		return H 
	
	
	
	
	
class ll_light():
	def __init__(self,ll):
		"""A minimalistic version of LL object for multiprocessing. Reduces the amount of information 
			transfered to the nodes"""
		self.e				=	ll.e
		self.h_e_val		=	ll.h_e_val
		self.h_2e_val		=	ll.h_2e_val
		self.h_z_val		=	ll.h_z_val
		self.h_2z_val		=	ll.h_2z_val
		self.h_ez_val		=	ll.h_ez_val
		self.GAR_1MA		=	ll.GAR_1MA
		self.args_v			=	ll.args_v
		self.args_d			=	ll.args_d
		self.GAR_1			=	ll.GAR_1
		self.AMA_1			=	ll.AMA_1
		self.re_obj_i		=	ll.re_obj_i
		self.re_obj_t		=	ll.re_obj_t
		self.dvarRE_input	=	ll.dvarRE_input
		self.ddvarRE_input	=	ll.ddvarRE_input
		self.e_RE			=	ll.e_RE
		self.re_obj_i_v		=	ll.re_obj_i_v
		self.re_obj_t_v		=	ll.re_obj_t_v
		self.dlnvRE			=	ll.dlnvRE
		self.ddlnvRE		=	ll.ddlnvRE
		self.varRE_input	=	ll.varRE_input
		self.vRE			=	ll.vRE
		self.lnvRE			=	ll.lnvRE
		self.dlnvRE_mu		=	ll.dlnvRE_mu
		self.ddlnvRE_mu_vRE	=	ll.ddlnvRE_mu_vRE
		



class g_obj_light():
	def __init__(self,g):
		"""A minimalistic version of g object for multiprocessing. Reduces the amount of information 
			transfered to the nodes"""

		self.DLL_e			=	g.DLL_e
		self.dLL_lnv		=	g.dLL_lnv
		self.dlnv_gamma		=	g.dlnv_gamma
		self.dlnv_psi		=	g.dlnv_psi
		self.dlnv_sigma_rho_G	=	g.dlnv_sigma_rho_G
		self.dlnv_sigma_lambda_G=	g.dlnv_sigma_lambda_G
		self.dlnv_sigma_beta_G	=	g.dlnv_sigma_beta_G
		self.dlnv_z_G		=	g.dlnv_z_G
		self.de_lambda		=	g.de_lambda
		self.de_rho			=	g.de_rho
		self.de_beta		=	g.de_beta
		self.de_lambda_RE	=	g.de_lambda_RE
		self.de_rho_RE		=	g.de_rho_RE
		self.de_beta_RE		=	g.de_beta_RE		
		self.dvRE_rho		=	g.dvRE_rho
		self.dvRE_lambda	=	g.dvRE_lambda
		self.dvRE_beta		=	g.dvRE_beta
		self.d_rho_input	=	g.d_rho_input
		self.d_lambda_input	=	g.d_lambda_input
		self.d_beta_input	=	g.d_beta_input



		
def T(x):
	if x is None:
		return None
	return x.T