#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .. import functions as fu
from . import calculus_functions as cf
from . import function
import numpy as np
import time

class gradient:

  def __init__(self,panel,callback):
    self.panel=panel
    self.callback=callback

  def arima_grad(self,k,x,ll,sign,pre, panel):
    if k==0:
      return None
    x = dotroll(panel, pre,k,sign,x,ll)
    extr_value=1e+100
    if np.max(np.abs(x))>extr_value:
      x[np.abs(x)>extr_value]=np.sign(x[np.abs(x)>extr_value])*extr_value
    return x*self.panel.a[3][0]

  def garch_arima_grad(self,ll,d,varname):
    panel=self.panel
    dvar_sigma=None
    if self.panel.pqdkm[4]>0 and not d is None: 			#eqs. 33-34
      x=cf.prod((ll.h_e_val,d))	
      dvar_sigma = dot(panel, ll.GAR_1MA,x,ll)
    return dvar_sigma


  def get(self,ll,DLL_e=None,dLL_var=None):


 
    (self.DLL_e, self.dLL_var)=(DLL_e, dLL_var)
    panel=self.panel
    incl=self.panel.included[3][0]
    
    u, e,h_e_val,var,h_val,v=ll.u, ll.e,ll.h_e_val,ll.var,ll.h_val,ll.v
    p,q,d,k,m=panel.pqdkm
    nW=panel.nW
    if DLL_e is None:
      dLL_var, DLL_e=function.gradient(ll,self.panel)
    self.X = reshape(panel.XIV)*panel.included[3][0]
    #ARIMA:
    de_rho=self.arima_grad(p,u,ll,-1,ll.AMA_1, panel)
    de_lambda=self.arima_grad(q,e,ll,-1,ll.AMA_1, panel)
    de_beta=-self.panel.arma_dot.dot(ll.AMA_1AR,self.X,ll)*panel.a[3][0]

    (self.de_rho,self.de_lambda,self.de_beta)=(de_rho,de_lambda,de_beta)		

    dvar_sigma_rho		=	self.garch_arima_grad(ll,	self.de_rho,		'rho')
    dvar_sigma_lambda	=	self.garch_arima_grad(ll,	self.de_lambda,	'lambda')
    dvar_sigma_beta	=	self.garch_arima_grad(ll,	self.de_beta,	'beta')


    (self.dvar_sigma_rho,self.dvar_sigma_lambda,self.dvar_sigma_beta)=(dvar_sigma_rho,dvar_sigma_lambda,dvar_sigma_beta)

    #GARCH:

    self.dvar_omega=dot(panel, ll.GAR_1,reshape(panel.W_a),ll)
    self.dvar_initvar = None
    if 'initvar' in ll.args.args_d:
      self.dvar_initvar = ll.GAR_1[0].reshape((1,panel.max_T,1))
      if not panel.options.EGARCH.value and ll.args.args_d['initvar'][0]<0:	
        self.dvar_initvar = -self.dvar_initvar
      
    (dvar_gamma, dvar_psi, dvar_z)=(None,None,None)	

    if m>0:
      dvar_gamma=self.arima_grad(k,var,ll,1,ll.GAR_1, panel)
      dvar_psi=self.arima_grad(m,h_val,ll,1,ll.GAR_1, panel)
      if not ll.h_z_val is None:
        dvar_z=fu.dot(ll.GAR_1MA,ll.h_z_val)

    (self.dvar_gamma, self.dvar_psi,self.dvar_z,self.dvar_z)=(dvar_gamma, dvar_psi, dvar_z, dvar_z)

    #LL


    #final derivatives:
    dLL_beta=cf.add((cf.prod((dvar_sigma_beta,dLL_var)),cf.prod((self.de_beta,DLL_e))),True)
    dLL_rho=cf.add((cf.prod((dvar_sigma_rho,dLL_var)),cf.prod((self.de_rho,DLL_e))),True)
    dLL_lambda=cf.add((cf.prod((dvar_sigma_lambda,dLL_var)),cf.prod((self.de_lambda,DLL_e))),True)
    dLL_gamma=cf.prod((dvar_gamma,dLL_var))
    dLL_psi=cf.prod((dvar_psi,dLL_var))
    dLL_omega=cf.prod((self.dvar_omega,dLL_var))
    dLL_initvar=cf.prod((self.dvar_initvar,dLL_var))
    dLL_z=cf.prod((self.dvar_z,dLL_var))


    G=cf.concat_marray((dLL_beta,dLL_rho,dLL_lambda,dLL_gamma,dLL_psi,dLL_omega, dLL_initvar,dLL_z))
    g=np.sum(G,0)
    #For debugging:
    #from .. import debug
    #print(debug.grad_debug(ll,panel,0.00001))
    #print(g)
    #if np.sum((g-gn)**2)>10000000:
    #	a=0
    #print(gn)
    #a=debug.grad_debug_detail(ll, panel, 0.00000001, 'LL', 'beta',0)
    #dLLeREn,deREn=debug.LL_calc_custom(ll, panel, 0.0000001)

    self.callback(perc = 0.08, text = '', task = 'gradient')
    



    return g, G


def dotroll(panel, pre,lags,sign,x,ll):
  (T,k) = x.shape
  x = x.reshape((1,T,k))  
  x = panel.arma_dot.dotroll(pre,lags,sign,x,ll)
  x = x.reshape((T,k)) 
  return x
  
  
def dot(panel, GAR_1MA,x,ll):
  (T,k) = x.shape
  x = x.reshape((1,T,k))  
  x = panel.arma_dot.dot(GAR_1MA,x,ll)
  x = x.reshape((T,k)) 
  return x
  
  
def reshape(x):
  N,T,k = x.shape
  x = x.reshape((T,k))
  return x