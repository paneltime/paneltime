#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def LL(panel,lnv,e_REsq, e_RE):
	incl=panel.a[3]
	
	LL_const=-0.5*np.log(2*np.pi)
	if panel.options.normal_GARCH.value>0:
		a,k=panel.options.GARCH_assist.value, panel.options.kurtosis_adj.value
		
		dlnv_pos=(lnv<1e+100)*(lnv>1e-100)
		lnv = incl*np.maximum(np.minimum(lnv,1e+100),1e-100)
		v=lnv
		v_inv = incl/(lnv+incl)	
		if panel.options.normal_GARCH.value==1:
			
			LL_full = LL_const-0.5*(incl*np.log(lnv+incl)+(1-k)*e_REsq*v_inv
									+ a* (np.abs(e_REsq-lnv)*v_inv)
									+ (k/3)* e_REsq**2*v_inv**2
									)				
	else:
		dlnv_pos=(lnv<100)*(lnv>-100)
		lnv = np.maximum(np.minimum(lnv,100),-100)
		v = np.exp(lnv)*incl
		v_inv = np.exp(-lnv)*incl		
		LL_full = LL_const-0.5*(lnv+(e_REsq)*v_inv)
	return LL_full,v,v_inv,dlnv_pos


def gradient(ll,panel):
	incl=panel.included[3]
	a,k=panel.options.GARCH_assist.value, panel.options.kurtosis_adj.value
	lnv,e_REsq,e_RE,v_inv=ll.lnv, ll.e_REsq, ll.e_RE,ll.v_inv 
	
	if panel.options.normal_GARCH.value:
		DLL_e   =-0.5*(	(1-k)*2*e_RE*v_inv	)
		dLL_lnv =-0.5*(	incl/(lnv+incl)-(1-k)*(e_REsq)*v_inv**2	)

		DLL_e +=-0.5*(		
			  a* 2*np.sign(e_REsq-lnv)*e_RE*v_inv
			+ (k/3)* 4*e_REsq*e_RE*v_inv**2
				 )
		dLL_lnv +=-0.5*(	
				- a* (np.sign(e_REsq-lnv)*v_inv)
						- a* (np.abs(e_REsq-lnv)*v_inv**2)
				- (k/3)* 2*e_REsq**2*v_inv**3
				 )
	else:	
		DLL_e=-(ll.e_RE*ll.v_inv)
		dLL_lnv=-0.5*(incl-(ll.e_REsq*ll.v_inv)*incl)	
	dLL_lnv*=ll.dlnv_pos*incl	
	DLL_e*=incl
	
	return dLL_lnv, DLL_e

def hessian(ll,panel):
	incl=panel.included[3]
	lnv,e_REsq,e_RE,v_inv=ll.lnv, ll.e_REsq, ll.e_RE,ll.v_inv 
	a,k=panel.options.GARCH_assist.value, panel.options.kurtosis_adj.value
	
	if panel.options.normal_GARCH.value:	
		d2LL_de2 	=-0.5*(	(1-k)*2*v_inv	)
		d2LL_dln_de =-0.5*(	-(1-k)*2*e_RE*v_inv**2)
		d2LL_dln2 	=-0.5*(-1*incl/(lnv+incl)**2+(1-k)*2*(e_REsq)*v_inv**3	)
		
		d2LL_de2 	+=-0.5*(		
				  a* 2*np.sign(e_REsq-lnv)*v_inv
				+ (k/3)* 12*e_REsq*v_inv**2
					 )
		d2LL_dln_de +=-0.5*(		
				- a* 2*np.sign(e_REsq-lnv)*e_RE*v_inv**2
				- (k/3)* 8*e_REsq*e_RE*v_inv**3
					 )
		d2LL_dln2 	+=-0.5*(	
					+ a* (np.sign(e_REsq-lnv)*v_inv**2)
							+ a* 2*(np.abs(e_REsq-lnv)*v_inv**3)
							+ a* (np.sign(e_REsq-lnv)*v_inv**2)
					+ (k/3)* 6*e_REsq**2*v_inv**4
					 )

	else:
		d2LL_de2=-ll.v_inv
		d2LL_dln_de=ll.e_RE*ll.v_inv
		d2LL_dln2=-0.5*ll.e_REsq*ll.v_inv
	d2LL_dln_de*=ll.dlnv_pos*incl
	d2LL_dln2*=ll.dlnv_pos*incl
	d2LL_de2*=incl
	
	return d2LL_de2, d2LL_dln_de, d2LL_dln2

