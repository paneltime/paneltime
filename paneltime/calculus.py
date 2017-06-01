#!/usr/bin/env python
# -*- coding: utf-8 -*-

import regprocs as rp
import numpy as np
import functions as fu
import time
import debug
import os
import multi_core

class gradient:
	
	def __init__(self,panel):
		self.panel=panel
		
	def arima_grad(self,k,x,sign=1,pre=None):
		if k==0:
			return None
		L=self.panel.L
		(N,T,m)=x.shape
		#L is "T x T" 
		#x is "N x T x 1"  
		#creates a  "k x T x N x 1": 
		x=np.array([np.dot(L[i],x) for i in range(k)])
		#reshapes to  "1 x T x N x k": 
		x=np.swapaxes(x,0,3)
		#reshapes to  "N x T x k": 
		x=np.swapaxes(x,1,2).reshape((N,T,k))
		if not pre is None:
			x=fu.dot(pre,x)
		if sign<0:
			x=x*sign
		return x

	def garch_arima_grad(self,ll,d):
		if self.panel.m>0 and not d is None:
			((N,T,k))=d.shape
			x=rp.prod((ll.h_e_val,d))
			dlnv_e_G=fu.dot(ll.GAR_1MA,x)
			mu=0
			if self.panel.N>1:
				mu=ll.args_d['mu']
			dlnv_e=dlnv_e_G+mu*(np.sum(x,1)/self.panel.T_arr).reshape((N,1,k))*self.panel.a#adds also the average inverted error ter
			return dlnv_e,dlnv_e_G
		else:
			return None,None

	def get(self,ll,return_G=False):
		panel=self.panel
		u,e,h_e_val,lnv_ARMA,h_val,v,v_inv=ll.u,ll.e,ll.h_e_val,ll.lnv_ARMA,ll.h_val,ll.v,ll.v_inv
		p,d,q,m,k,nW=panel.p,panel.d,panel.q,panel.m,panel.k,panel.nW

		#ARIMA:
		de_rho=self.arima_grad(p,u,-1,ll.AMA_1)
		de_lambda=self.arima_grad(q,e,-1,ll.AMA_1)
		de_beta=-fu.dot(ll.AMA_1AR,panel.X)
		(ll.de_rho,ll.de_lambda,ll.de_beta)=(de_rho,de_lambda,de_beta)

		ll.de_rho_RE,ll.de_lambda_RE,ll.de_beta_RE=rp.dRE(ll,panel,de_rho,ll.e,'rho'),rp.dRE(ll,panel,de_lambda,ll.e,'lambda'),rp.dRE(ll,panel,de_beta,ll.e,'beta')
		


		dlnv_e_rho,		dlnv_e_rho_G	=	self.garch_arima_grad(ll,de_rho)
		dlnv_e_lambda, 	dlnv_e_lambda_G	=	self.garch_arima_grad(ll,de_lambda)
		dlnv_e_beta,	dlnv_e_beta_G	=	self.garch_arima_grad(ll,de_beta)

		(ll.dlnv_e_rho,ll.dlnv_e_lambda,ll.dlnv_e_beta)=(dlnv_e_rho,dlnv_e_lambda,dlnv_e_beta)
		(ll.dlnv_e_rho_G,ll.dlnv_e_lambda_G,ll.dlnv_e_beta_G)=(dlnv_e_rho_G,dlnv_e_lambda_G,dlnv_e_beta_G)

		#GARCH:
		(dlnv_gamma, dlnv_psi, dlnv_mu, dlnv_z_G, dlnv_z)=(None,None,None,None,None)
		if panel.m>0:
			dlnv_gamma=self.arima_grad(k,lnv_ARMA,1,ll.GAR_1)
			dlnv_psi=self.arima_grad(m,h_val,1,ll.GAR_1)
			dlnv_z_G=fu.dot(ll.GAR_1MA,ll.h_z_val)
			(N,T,k)=dlnv_z_G.shape
			if panel.N>1:
				mu=ll.args_d['mu']
				dlnv_mu=ll.avg_h
			else:
				mu=0
				dlnv_mu=None
			dlnv_z=dlnv_z_G+(mu*(np.sum(ll.h_z_val,1)/panel.T_arr)).reshape(N,1,1)


		(ll.dlnv_gamma, ll.dlnv_psi,ll.dlnv_mu,ll.dlnv_z_G,ll.dlnv_z)=(dlnv_gamma, dlnv_psi, dlnv_mu, dlnv_z_G, dlnv_z)

		#LL

		DLL_e=-(ll.e_RE*ll.v_inv)*panel.included
		dLL_lnv=-0.5*(panel.included-(ll.e_REsq*v_inv)*panel.included)
		(ll.DLL_e, ll.dLL_lnv)=(DLL_e, dLL_lnv)


		#final derivatives:
		dLL_beta=rp.add((rp.prod((dlnv_e_beta,dLL_lnv)),rp.prod((ll.de_beta_RE,DLL_e))),True)
		dLL_rho=rp.add((rp.prod((dlnv_e_rho,dLL_lnv)),rp.prod((ll.de_rho_RE,DLL_e))),True)
		dLL_lambda=rp.add((rp.prod((dlnv_e_lambda,dLL_lnv)),rp.prod((ll.de_lambda_RE,DLL_e))),True)
		dLL_gamma=rp.prod((dlnv_gamma,dLL_lnv))
		dLL_psi=rp.prod((dlnv_psi,dLL_lnv))
		dLL_omega=rp.prod((panel.W_a,dLL_lnv))
		dLL_mu=rp.prod((ll.dlnv_mu,dLL_lnv))
		dLL_z=rp.prod((ll.dlnv_z,dLL_lnv))

		G=rp.concat_marray((dLL_beta,dLL_rho,dLL_lambda,dLL_gamma,dLL_psi,dLL_omega,dLL_mu,dLL_z))
		g=np.sum(np.sum(G,0),0)
		#print (g)
		#gn=debug.grad_debug(panel,ll.args_v,0.0000001)#debugging
		#debug.LL_debug_detail(ll,panel,0.0001)
		if return_G:
			return  g,G
		else:
			return g


class hessian:
	def __init__(self,panel):
		self.panel=panel
		self.its=0
		self.sec_deriv=self.set_mp_strings()
		
	
	def get(self,ll,mp):	
		if mp is None:
			return self.hessian(ll)
		else:
			return self.hessian_mp(ll,mp)

	def hessian(self,ll):
		panel=self.panel
		tic=time.clock()
		GARM=rp.ARMA_product(ll.GAR_1,panel.L,panel.m)
		GARK=rp.ARMA_product(ll.GAR_1,panel.L,panel.k)

		d2lnv_gamma2		=   rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_gamma,						ll.dLL_lnv,  transpose=True)
		d2lnv_gamma_psi		=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_psi,						ll.dLL_lnv)

		d2lnv_gamma_rho		=	rp.dd_func_lags(panel,ll,GARK,	ll.dlnv_e_rho_G,					ll.dLL_lnv)
		d2lnv_gamma_lambda	=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_e_lambda_G,					ll.dLL_lnv)
		d2lnv_gamma_beta	=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_e_beta_G,					ll.dLL_lnv)
		d2lnv_gamma_z		=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_z_G,						ll.dLL_lnv)

		d2lnv_psi_rho		=	rp.dd_func_lags(panel,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_rho)),	ll.dLL_lnv)
		d2lnv_psi_lambda	=	rp.dd_func_lags(panel,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_lambda)),	ll.dLL_lnv)
		d2lnv_psi_beta		=	rp.dd_func_lags(panel,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_beta)),	ll.dLL_lnv)
		d2lnv_psi_z			=	rp.dd_func_lags(panel,ll,GARM, 	ll.h_z_val,							ll.dLL_lnv)

		AMAq=-rp.ARMA_product(ll.AMA_1,panel.L,panel.q)
		d2lnv_lambda2,		d2e_lambda2		=	rp.dd_func_lags_mult(panel,ll,AMAq,	ll.de_lambda,	ll.de_lambda,	'lambda',	'lambda', transpose=True)
		d2lnv_lambda_rho,	d2e_lambda_rho	=	rp.dd_func_lags_mult(panel,ll,AMAq,	ll.de_lambda,	ll.de_rho,		'lambda',	'rho' )
		d2lnv_lambda_beta,	d2e_lambda_beta	=	rp.dd_func_lags_mult(panel,ll,AMAq,	ll.de_lambda,	ll.de_beta,		'lambda',	'beta')

		AMAp=-rp.ARMA_product(ll.AMA_1,panel.L,panel.p)
		d2lnv_rho_beta,		d2e_rho_beta	=	rp.dd_func_lags_mult(panel,ll,AMAp,	ll.de_rho,		ll.de_beta,		'rho',		'beta', de_zeta_u=-panel.X)
		
		d2lnv_mu_rho,d2lnv_mu_lambda,d2lnv_mu_beta,d2lnv_mu_z,mu=None,None,None,None,None
		if panel.N>1:
			d2lnv_mu_rho			=	rp.dd_func_lags(panel,ll,None, 		rp.prod((ll.h_e_val,ll.de_rho)),	ll.dLL_lnv, 	addavg=1) 
			d2lnv_mu_lambda			=	rp.dd_func_lags(panel,ll,None, 		rp.prod((ll.h_e_val,ll.de_lambda)),	ll.dLL_lnv, 	addavg=1) 
			d2lnv_mu_beta			=	rp.dd_func_lags(panel,ll,None, 		rp.prod((ll.h_e_val,ll.de_beta)),	ll.dLL_lnv, 	addavg=1) 
			d2lnv_mu_z				=	rp.dd_func_lags(panel,ll,None, 		ll.h_z_val,							ll.dLL_lnv, 	addavg=1) 
			mu=ll.args_d['mu']
	
		d2lnv_z2				=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, ll.h_2z_val,						ll.dLL_lnv, 	addavg=mu) 
		d2lnv_z_rho				=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_rho)),	ll.dLL_lnv, 	addavg=mu) 
		d2lnv_z_lambda			=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_lambda)),ll.dLL_lnv, 	addavg=mu) 
		d2lnv_z_beta			=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_beta)),	ll.dLL_lnv, 	addavg=mu) 
		
		d2lnv_rho2,	d2e_rho2	=	rp.dd_func_lags_mult(panel,ll,	None,	ll.de_rho,		ll.de_rho,		'rho',		'rho' )
		d2lnv_beta2,d2e_beta2	=	rp.dd_func_lags_mult(panel,ll,	None,	ll.de_beta,		ll.de_beta,		'beta',		'beta')


		d2LL_de2=-ll.v_inv*panel.included
		d2LL_dln_de=ll.e_RE*ll.v_inv*panel.included
		d2LL_dln2=-0.5*ll.e_REsq*ll.v_inv*panel.included
		(de_rho_RE,de_lambda_RE,de_beta_RE)=(ll.de_rho_RE,ll.de_lambda_RE,ll.de_beta_RE)
		(dlnv_e_rho,dlnv_e_lambda,dlnv_e_beta)=(ll.dlnv_e_rho,ll.dlnv_e_lambda,ll.dlnv_e_beta)
		(dlnv_gamma,dlnv_psi)=(ll.dlnv_gamma,ll.dlnv_psi)
		(dlnv_mu,dlnv_z)=(ll.dlnv_mu, ll.dlnv_z)		

		#Final:
		D2LL_beta2			=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_beta_RE,		dlnv_e_beta, 	dlnv_e_beta,	d2e_beta2, 					d2lnv_beta2)
		D2LL_beta_rho		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_rho_RE,		dlnv_e_beta, 	dlnv_e_rho,		T(d2e_rho_beta), 		T(d2lnv_rho_beta))
		D2LL_beta_lambda	=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_lambda_RE,	dlnv_e_beta, 	dlnv_e_lambda,	T(d2e_lambda_beta), 	T(d2lnv_lambda_beta))
		D2LL_beta_gamma		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_beta))
		D2LL_beta_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_beta))
		D2LL_beta_omega		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	panel.W_a,		None, 					None)
		D2LL_beta_mu		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	dlnv_mu,		None, 					T(d2lnv_mu_beta))
		D2LL_beta_z			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	dlnv_z,			None, 					T(d2lnv_z_beta))
		
		D2LL_rho2			=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_rho_RE,		dlnv_e_rho, 	dlnv_e_rho,		d2e_rho2, 					d2lnv_rho2)
		D2LL_rho_lambda		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_lambda_RE,	dlnv_e_rho, 	dlnv_e_lambda,	T(d2e_lambda_rho), 		T(d2lnv_lambda_rho))
		D2LL_rho_gamma		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_rho))	
		D2LL_rho_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_rho))
		D2LL_rho_omega		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	panel.W_a,		None, 					None)
		D2LL_rho_mu			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	dlnv_mu,		None, 					T(d2lnv_mu_rho))
		D2LL_rho_z			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	dlnv_z,			None, 					T(d2lnv_z_rho))
		
		D2LL_lambda2		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	de_lambda_RE,	dlnv_e_lambda, 	dlnv_e_lambda,	T(d2e_lambda2), 		T(d2lnv_lambda2))
		D2LL_lambda_gamma	=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_lambda))
		D2LL_lambda_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_lambda))
		D2LL_lambda_omega	=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	panel.W_a,		None, 					None)
		D2LL_lambda_mu		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	dlnv_mu,		None, 					T(d2lnv_mu_lambda))
		D2LL_lambda_z		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	dlnv_z,			None, 					T(d2lnv_z_lambda))
		
		D2LL_gamma2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma2))
		D2LL_gamma_psi		=	rp.dd_func(None,		None,			d2LL_dln2,	None,			None,			ll.dlnv_gamma, 	ll.dlnv_psi,		None, 					d2lnv_gamma_psi)
		D2LL_gamma_omega	=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	panel.W_a,		None, 					None)
		D2LL_gamma_mu		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	dlnv_mu,		None, 					None)
		D2LL_gamma_z		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	dlnv_z,			None, 					d2lnv_gamma_z)
		
		D2LL_psi2			=	rp.dd_func(None,		None,			d2LL_dln2,	None,			None,			ll.dlnv_psi, 		ll.dlnv_psi,		None, 					None)
		D2LL_psi_omega		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		panel.W_a,		None, 					None)
		D2LL_psi_mu			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		dlnv_mu,		None, 					None)
		D2LL_psi_z			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		dlnv_z,			None, 					d2lnv_psi_z)
		
		D2LL_omega2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		panel.W_a,		None, 					None)
		D2LL_omega_mu		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_mu,		None, 					None)
		D2LL_omega_z		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_z,			None, 					None)
		
		D2LL_mu2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_mu,		None, 					None)
		D2LL_mu_z			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_z,			None, 					d2lnv_mu_z)
		
		D2LL_z2				=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_z, 		dlnv_z,			None, 					d2lnv_z2)

		H= [[D2LL_beta2,			D2LL_beta_rho,		D2LL_beta_lambda,		D2LL_beta_gamma,	D2LL_beta_psi,		D2LL_beta_omega,	D2LL_beta_mu,	D2LL_beta_z		],
	        [T(D2LL_beta_rho),		D2LL_rho2,			D2LL_rho_lambda,		D2LL_rho_gamma,		D2LL_rho_psi,		D2LL_rho_omega,		D2LL_rho_mu,	D2LL_rho_z			],
	        [T(D2LL_beta_lambda),	T(D2LL_rho_lambda),	D2LL_lambda2,			D2LL_lambda_gamma,	D2LL_lambda_psi,	D2LL_lambda_omega,	D2LL_lambda_mu,	D2LL_lambda_z		],
	        [T(D2LL_beta_gamma),	T(D2LL_rho_gamma),	T(D2LL_lambda_gamma),	D2LL_gamma2,		D2LL_gamma_psi,		D2LL_gamma_omega, 	D2LL_gamma_mu,	D2LL_gamma_z		],
	        [T(D2LL_beta_psi),		T(D2LL_rho_psi),	T(D2LL_lambda_psi),		T(D2LL_gamma_psi),	D2LL_psi2,			D2LL_psi_omega, 	D2LL_psi_mu,	D2LL_psi_z			],
	        [T(D2LL_beta_omega),	T(D2LL_rho_omega),	T(D2LL_lambda_omega),	T(D2LL_gamma_omega),T(D2LL_psi_omega),	D2LL_omega2, 		D2LL_omega_mu,	D2LL_omega_z		], 
	        [T(D2LL_beta_mu),		T(D2LL_rho_mu),		T(D2LL_lambda_mu),		T(D2LL_gamma_mu),	T(D2LL_psi_mu),		T(D2LL_omega_mu), 	D2LL_mu2,		D2LL_mu_z			],
	        [T(D2LL_beta_z),		T(D2LL_rho_z),		T(D2LL_lambda_z),		T(D2LL_gamma_z),	T(D2LL_psi_z),		T(D2LL_omega_z), 	D2LL_mu_z,		D2LL_z2				]]

		H=rp.concat_matrix(H)
		#for debugging:
		#Hn=debug.hess_debug(panel,ll.args_v,0.000001)#debugging
		#debug.LL_debug_detail(self,ll,0.0000001)
		#print (time.clock()-tic)
		self.its+=1
		
		W=H*1
		for i in range(len(W)):
			W[i,i]=W[i,i]+(W[i,i]==0)
		W=np.linalg.inv(W)
		WH=np.dot(W,H)	
		HWH=np.dot(H.T,WH)
		return H
	


	def second_derivatives_mp(self,ll,mp):
		panel=self.panel
		mp.send_dict({'ll':ll_light(ll)},'dynamic dictionary')	
		d=mp.execute(self.sec_deriv)

		d['d2LL_de2']=-ll.v_inv*panel.included
		d['d2LL_dln_de']=ll.e_RE*ll.v_inv*panel.included
		d['d2LL_dln2']=-0.5*ll.e_REsq*ll.v_inv*panel.included
		(d['de_rho_RE'],d['de_lambda_RE'],d['de_beta_RE'])=(ll.de_rho_RE,ll.de_lambda_RE,ll.de_beta_RE)
		(d['dlnv_e_rho'],d['dlnv_e_lambda'],d['dlnv_e_beta'])=(ll.dlnv_e_rho,ll.dlnv_e_lambda,ll.dlnv_e_beta)
		(d['dlnv_gamma'],d['dlnv_psi'])=(ll.dlnv_gamma,ll.dlnv_psi)

		(d['dlnv_mu'], d['dlnv_z'])=(ll.dlnv_mu, ll.dlnv_z)		

		return d
	
	def set_mp_strings(self):
		#these are all "k x T x T" matrices:
		evalstr=[]		
		#strings are evaluated for the code to be compatible with multi core proccessing
		evalstr.append("""
			                    GARM=rp.ARMA_product(ll.GAR_1,panel.L,panel.m)
			                    GARK=rp.ARMA_product(ll.GAR_1,panel.L,panel.k)
	
			                    d2lnv_gamma2		=   rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_gamma,						ll.dLL_lnv,  transpose=True)
			                    d2lnv_gamma_psi		=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_psi,						ll.dLL_lnv)
	
			                    d2lnv_gamma_rho		=	rp.dd_func_lags(panel,ll,GARK,	ll.dlnv_e_rho_G,					ll.dLL_lnv)
			                    d2lnv_gamma_lambda	=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_e_lambda_G,					ll.dLL_lnv)
			                    d2lnv_gamma_beta	=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_e_beta_G,					ll.dLL_lnv)
			                    d2lnv_gamma_z		=	rp.dd_func_lags(panel,ll,GARK, 	ll.dlnv_z_G,						ll.dLL_lnv)
	
			                    d2lnv_psi_rho		=	rp.dd_func_lags(panel,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_rho)),	ll.dLL_lnv)
			                    d2lnv_psi_lambda	=	rp.dd_func_lags(panel,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_lambda)),	ll.dLL_lnv)
			                    d2lnv_psi_beta		=	rp.dd_func_lags(panel,ll,GARM, 	rp.prod((ll.h_e_val,ll.de_beta)),	ll.dLL_lnv)
			                    d2lnv_psi_z			=	rp.dd_func_lags(panel,ll,GARM, 	ll.h_z_val,							ll.dLL_lnv)
			                    GARM=0#Releases memory
			                    GARK=0#Releases memory
			                    """)
		#ARCH:
		evalstr.append("""
			                    AMAq=-rp.ARMA_product(ll.AMA_1,panel.L,panel.q)
			                    d2lnv_lambda2,		d2e_lambda2		=	rp.dd_func_lags_mult(panel,ll,AMAq,	ll.de_lambda,	ll.de_lambda,	'lambda',	'lambda', transpose=True)
			                    d2lnv_lambda_rho,	d2e_lambda_rho	=	rp.dd_func_lags_mult(panel,ll,AMAq,	ll.de_lambda,	ll.de_rho,		'lambda',	'rho' )
			                    d2lnv_lambda_beta,	d2e_lambda_beta	=	rp.dd_func_lags_mult(panel,ll,AMAq,	ll.de_lambda,	ll.de_beta,		'lambda',	'beta')
			                    AMAq=0#Releases memory
			                    """)
		evalstr.append("""		
	
			                    AMAp=-rp.ARMA_product(ll.AMA_1,panel.L,panel.p)
			                    d2lnv_rho_beta,		d2e_rho_beta	=	rp.dd_func_lags_mult(panel,ll,AMAp,	ll.de_rho,		ll.de_beta,		'rho',		'beta', de_zeta_u=-panel.X)
	
			                    d2lnv_mu_rho,d2lnv_mu_lambda,d2lnv_mu_beta,d2lnv_mu_z,mu=None,None,None,None,None
			                    if panel.N>1:
	
			                        d2lnv_mu_rho			=	rp.dd_func_lags(panel,ll,None, 		rp.prod((ll.h_e_val,ll.de_rho)),	ll.dLL_lnv, 	addavg=1) 
			                        d2lnv_mu_lambda			=	rp.dd_func_lags(panel,ll,None, 		rp.prod((ll.h_e_val,ll.de_lambda)),	ll.dLL_lnv, 	addavg=1) 
			                        d2lnv_mu_beta			=	rp.dd_func_lags(panel,ll,None, 		rp.prod((ll.h_e_val,ll.de_beta)),	ll.dLL_lnv, 	addavg=1) 
			                        d2lnv_mu_z				=	rp.dd_func_lags(panel,ll,None, 		ll.h_z_val,							ll.dLL_lnv, 	addavg=1) 
			                        mu=ll.args_d['mu']
	
			                    d2lnv_z2				=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, ll.h_2z_val,						ll.dLL_lnv, 	addavg=mu) 
			                    d2lnv_z_rho				=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_rho)),	ll.dLL_lnv, 	addavg=mu) 
			                    d2lnv_z_lambda			=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_lambda)),ll.dLL_lnv, 	addavg=mu) 
			                    d2lnv_z_beta			=	rp.dd_func_lags(panel,ll,ll.GAR_1MA, rp.prod((ll.h_ez_val,ll.de_beta)),	ll.dLL_lnv, 	addavg=mu) 
	
			                    d2lnv_rho2,	d2e_rho2	=	rp.dd_func_lags_mult(panel,ll,	None,	ll.de_rho,		ll.de_rho,		'rho',		'rho' )
			                    AMAp=0#Releases memory
			                    """)
		evalstr.append("""	
	
			                    d2lnv_beta2,d2e_beta2	=	rp.dd_func_lags_mult(panel,ll,	None,	ll.de_beta,		ll.de_beta,		'beta',		'beta')
			                    """)
	
		return multi_core.format_args_array(evalstr)	



	def hessian_mp(self,ll,mp):
		panel=self.panel
		tic=time.clock()
		#return debug.hessian_debug(self,args):
		d=self.second_derivatives_mp(ll,mp)
		#Final:
		evalstr="""
D2LL_beta2			=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_beta_RE,		dlnv_e_beta, 	dlnv_e_beta,	d2e_beta2, 					d2lnv_beta2)
D2LL_beta_rho		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_rho_RE,		dlnv_e_beta, 	dlnv_e_rho,		T(d2e_rho_beta), 		T(d2lnv_rho_beta))
D2LL_beta_lambda	=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	de_lambda_RE,	dlnv_e_beta, 	dlnv_e_lambda,	T(d2e_lambda_beta), 	T(d2lnv_lambda_beta))
D2LL_beta_gamma		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_beta))
D2LL_beta_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_beta))
D2LL_beta_omega		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	panel.W_a,		None, 					None)
D2LL_beta_mu		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	dlnv_mu,		None, 					T(d2lnv_mu_beta))
D2LL_beta_z			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_beta_RE, 	None,			dlnv_e_beta, 	dlnv_z,			None, 					T(d2lnv_z_beta))

D2LL_rho2			=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_rho_RE,		dlnv_e_rho, 	dlnv_e_rho,		d2e_rho2, 					d2lnv_rho2)
D2LL_rho_lambda		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		de_lambda_RE,	dlnv_e_rho, 	dlnv_e_lambda,	T(d2e_lambda_rho), 		T(d2lnv_lambda_rho))
D2LL_rho_gamma		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_rho))	
D2LL_rho_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_rho))
D2LL_rho_omega		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	panel.W_a,		None, 					None)
D2LL_rho_mu			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	dlnv_mu,		None, 					T(d2lnv_mu_rho))
D2LL_rho_z			=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_rho_RE, 		None,			dlnv_e_rho, 	dlnv_z,			None, 					T(d2lnv_z_rho))

D2LL_lambda2		=	rp.dd_func(d2LL_de2,	d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	de_lambda_RE,	dlnv_e_lambda, 	dlnv_e_lambda,	T(d2e_lambda2), 		T(d2lnv_lambda2))
D2LL_lambda_gamma	=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma_lambda))
D2LL_lambda_psi		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	ll.dlnv_psi,		None, 					T(d2lnv_psi_lambda))
D2LL_lambda_omega	=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	panel.W_a,		None, 					None)
D2LL_lambda_mu		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	dlnv_mu,		None, 					T(d2lnv_mu_lambda))
D2LL_lambda_z		=	rp.dd_func(None,		d2LL_dln_de,	d2LL_dln2,	de_lambda_RE, 	None,			dlnv_e_lambda, 	dlnv_z,			None, 					T(d2lnv_z_lambda))

D2LL_gamma2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	ll.dlnv_gamma,		None, 					T(d2lnv_gamma2))
D2LL_gamma_psi		=	rp.dd_func(None,		None,			d2LL_dln2,	None,			None,			ll.dlnv_gamma, 	ll.dlnv_psi,		None, 					d2lnv_gamma_psi)
D2LL_gamma_omega	=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	panel.W_a,		None, 					None)
D2LL_gamma_mu		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	dlnv_mu,		None, 					None)
D2LL_gamma_z		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_gamma, 	dlnv_z,			None, 					d2lnv_gamma_z)

D2LL_psi2			=	rp.dd_func(None,		None,			d2LL_dln2,	None,			None,			ll.dlnv_psi, 		ll.dlnv_psi,		None, 					None)
D2LL_psi_omega		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		panel.W_a,		None, 					None)
D2LL_psi_mu			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		dlnv_mu,		None, 					None)
D2LL_psi_z			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			ll.dlnv_psi, 		dlnv_z,			None, 					d2lnv_psi_z)

D2LL_omega2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		panel.W_a,		None, 					None)
D2LL_omega_mu		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_mu,		None, 					None)
D2LL_omega_z		=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			panel.W_a, 		dlnv_z,			None, 					None)

D2LL_mu2			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_mu,		None, 					None)
D2LL_mu_z			=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_mu, 		dlnv_z,			None, 					d2lnv_mu_z)

D2LL_z2				=	rp.dd_func(None,		None,			d2LL_dln2,	None, 			None,			dlnv_z, 		dlnv_z,			None, 					d2lnv_z2)
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
		H=rp.concat_matrix(H)
		#for debugging:
		#Hn=debug.hess_debug(panel,ll.args_v,0.000001)#debugging
		#H_debug=hessian(self, ll)
		#debug.LL_debug_detail(self,ll,0.0000001)
		print (time.clock()-tic)
		self.its+=1
		return H 
	
	
	
	
	
class ll_light():
	def __init__(self,ll):
		"""A minimalistic version of LL object for multiprocessing. Reduces the amount of information 
			transfered to the nodes"""
		if hasattr(ll,'vLSDV'):
			self.vLSDV			=	ll.vLSDV
			self.theta			=	ll.theta
			if not hasattr(ll,'deFE'):
				raise RuntimeError('the method gradient, from this module, must precede hessian in the code')
			self.deFE			=	ll.deFE
			self.dvLSDV			=	ll.dvLSDV
			self.dtheta			=	ll.dtheta
		self.DLL_e			=	ll.DLL_e
		self.e				=	ll.e
		self.h_e_val		=	ll.h_e_val
		self.h_2e_val		=	ll.h_2e_val
		self.h_z_val		=	ll.h_z_val
		self.h_2z_val		=	ll.h_2z_val
		self.h_ez_val		=	ll.h_ez_val
		self.GAR_1MA		=	ll.GAR_1MA
		self.dLL_lnv		=	ll.dLL_lnv
		self.args_v			=	ll.args_v
		self.args_d			=	ll.args_d
		self.GAR_1			=	ll.GAR_1
		self.AMA_1			=	ll.AMA_1
		self.dlnv_gamma		=	ll.dlnv_gamma
		self.dlnv_psi		=	ll.dlnv_psi
		self.dlnv_e_rho_G	=	ll.dlnv_e_rho_G
		self.dlnv_e_lambda_G=	ll.dlnv_e_lambda_G
		self.dlnv_e_beta_G	=	ll.dlnv_e_beta_G
		self.dlnv_z_G		=	ll.dlnv_z_G
		self.de_lambda		=	ll.de_lambda
		self.de_rho			=	ll.de_rho
		self.de_beta		=	ll.de_beta



		
def T(x):
	if x is None:
		return None
	return x.T



