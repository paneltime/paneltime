#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object


from .. import functions as fu
from . import function
from pathlib import Path
import os
import numpy.ctypeslib as npct
import ctypes as ct
p = os.path.join(Path(__file__).parent.absolute(),'cfunctions')
if os.name=='nt':
  cfunct = npct.load_library('ctypes.dll',p)
else:
  cfunct = npct.load_library('ctypes.so',p)
import numpy as np
import traceback
import time


CDPT = ct.POINTER(ct.c_double) 
CIPT = ct.POINTER(ct.c_uint) 



class LL:
  """Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
  that store dynamic variables that depend on the \n
  If args is a dictionary, the ARMA-GARCH orders are 
  determined from the dictionary. If args is a vector, the ARMA-GARCH order needs to be consistent
  with the  panel object
  """
  def __init__(self,args,panel,constraints=None,print_err=False):
    self.err_msg=''
    self.errmsg_h=''


    self.args=panel.args.create_args(args,panel,constraints)
    self.h_err=""
    self.LL=None
    #self.LL=self.LL_calc(panel)
    try:
      self.LL=self.LL_calc(panel)
      if np.isnan(self.LL):
        self.LL=None						
    except Exception as e:
      if print_err:
        traceback.print_exc()
        print(self.errmsg_h)



  def LL_calc(self,panel):
    X= reshape(panel.XIV)
    T, k = X.shape
    incl = panel.included[3][0]
    self.set_var_bounds(panel)
    
    W_a = reshape(panel.W_a)
    
    G = fu.dot(W_a, self.args.args_d['omega'])
    if 'initvar' in self.args.args_d:
      G[0] += abs(self.args.args_d['initvar'][0])
    else:
      G[0] += panel.args.init_var
    
    #Idea for IV: calculate Z*u throughout. Mazimize total sum of LL. 
    u = reshape(panel.Y)-fu.dot(X,self.args.args_d['beta'])


    matrices=self.arma_calc(panel, u, self.h_add, G)
    if matrices is None:
      return None		
    AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h=matrices

    #NOTE: self.h_val itself is also set in ctypes.cpp/ctypes.c. If you change self.h_val below, you need to 
    #change it in the c-scripts too. self.h_val must be calcualted below as well for later calulcations. 
    if panel.options.EGARCH.value==0:
      esq =(e**2+(e==0)*1e-18) 
      nd =1
      self.h_val, self.h_e_val, self.h_2e_val = (e**2+self.h_add)*incl, nd*2*e*incl, nd*2*incl
      self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None	#possibility of additional parameter in sq res function		
    else:
      minesq = 1e-20
      esq =np.maximum(e**2,minesq)
      nd = e**2>minesq		

      self.h_val, self.h_e_val, self.h_2e_val = np.log(esq+self.h_add)*incl, 2*incl*e/(esq+self.h_add), incl*2/(esq+self.h_add) - incl*2*e**2/(esq+self.h_add)**2
      self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None	#possibility of additional parameter in sq res function		


    if True:#debug
      from .. import debug
      if np.any(h!=self.h_val):
        print('the h calculated in the c function and the self.h_val calcualted here do not match')
      debug.test_c_armas(u, var, e, panel, self, G)

    LL_full,v,v_inv,self.dvar_pos=function.LL(panel,var,esq, e, self.minvar, self.maxvar)
    LL=np.sum(LL_full*incl)
    self.LL_all=np.sum(LL_full)
    self.add_variables(panel,matrices, u, var, v, G,e,esq,v_inv,LL_full)
    if abs(LL)>1e+100: 
      return None				
    return LL

  def set_var_bounds(self, panel):
    if panel.options.EGARCH.value==0:
      self.minvar = 0.01*panel.args.init_var
      self.maxvar = 1000*panel.args.init_var
      self.h_add = panel.args.init_var
    else:
      self.minvar = -100
      self.maxvar = 100
      self.h_add = 0.1
      
  def add_variables(self,panel,matrices,u, var,v,G,e,esq,v_inv,LL_full):
    self.v_inv05=v_inv**0.5
    self.e_norm=e*self.v_inv05	
    self.u     = u
    self.var,  self.v,    self.LL_full = var,       v,    LL_full
    self.G=G
    self.e=e
    self.esq=esq
    self.v_inv=v_inv

  def arma_calc(self,panel, u, h_add, G):
    matrices =set_garch_arch(panel,self.args.args_d, u, h_add, G)
    if matrices is None:
      return None		
    self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA, self.e, self.var, self.h = matrices
    self.AMA_dict={'AMA_1':None,'AMA_1AR':None,'GAR_1':None,'GAR_1MA':None}		
    return matrices

def reshape(x):
  N,T,k = x.shape
  x = x.reshape((T,k))
  return x
   


def set_garch_arch(panel,args,u, h_add, G):
  """Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
  the diagonal band"""
  T, _ = u.shape
  rho=np.insert(-args['rho'],0,1)
  psi=args['psi']
  psi=np.insert(args['psi'],0,0) 

  AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h=(
          np.append([1],np.zeros(T-1)),
                np.zeros(T),
                np.append([1],np.zeros(T-1)),
                np.zeros(T),
                np.zeros((T,1)),
                np.zeros((T,1)),
                np.zeros((T,1))
        )



  lmbda = args['lambda']
  gmma = -args['gamma']
  
  parameters = np.array(( 1 , T , 
                  len(lmbda), len(rho), len(gmma), len(psi), 
                  panel.options.EGARCH.value, panel.tot_lost_obs, 
                  h_add))

  cfunct.armas(parameters.ctypes.data_as(CIPT), 
                     lmbda.ctypes.data_as(CDPT), rho.ctypes.data_as(CDPT),
                                                  gmma.ctypes.data_as(CDPT), psi.ctypes.data_as(CDPT),
                                                  AMA_1.ctypes.data_as(CDPT), AMA_1AR.ctypes.data_as(CDPT),
                                                  GAR_1.ctypes.data_as(CDPT), GAR_1MA.ctypes.data_as(CDPT),
                                                  u.ctypes.data_as(CDPT), 
                                                  e.ctypes.data_as(CDPT), 
                                                  var.ctypes.data_as(CDPT),
                                                  h.ctypes.data_as(CDPT),
                                                  G.ctypes.data_as(CDPT)
                                                  )		


  r=[]
  #Creating nympy arrays with name properties. 
  for i in ['AMA_1','AMA_1AR','GAR_1','GAR_1MA']:
    r.append((locals()[i],i))
  for i in ['e', 'var', 'h']:
    r.append(locals()[i])

  return r
  

