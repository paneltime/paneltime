#!/usr/bin/env python
# -*- coding: utf-8 -*-

#calculates the arma matrices for GARCH and ARIMA

from ctypes import create_unicode_buffer
from pathlib import Path
import numpy as np
import os
from .. import cfunctions


def set_garch_arch(panel,args,u, h_add, G):
  """Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
  the diagonal band"""
  N, T, _ = u.shape
  rho = round( np.insert(-args['rho'],0,1), panel)
  psi = args['psi']
  psi = round( np.insert(args['psi'],0,0), panel)



  lmbda = round( args['lambda'], panel)
  gmma =  round( -args['gamma'], panel)
  
  
  parameters = np.array(( N , T , 
                  len(lmbda), len(rho), len(gmma), len(psi), 
                  panel.options.EGARCH.value, panel.lost_obs, 
                  h_add))
  AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h = inv_c(parameters, lmbda, rho, gmma, psi, N, T, u, G, panel.T_arr)

  r=[]
  #Creating nympy arrays with name properties. 
  for i in ['AMA_1','AMA_1AR','GAR_1','GAR_1MA']:
    m = round(locals()[i], panel)
    r.append((m,i))
  for i in ['e', 'var', 'h']:
    r.append(locals()[i])

  return r

def inv_c(parameters,lmbda, rho,gmma, psi, N, T, u, G, T_arr):

  AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h=(
  np.append([1],np.zeros(T-1)),
        np.zeros(T),
        np.append([1],np.zeros(T-1)),
        np.zeros(T),
        np.zeros((N,T,1)),
        np.zeros((N,T,1)),
        np.zeros((N,T,1))
    )
 
  T_arr = np.array(T_arr.flatten(),dtype = float)
  cfunctions.armas(parameters, lmbda, rho, gmma, psi, 
                    AMA_1, AMA_1AR, GAR_1, GAR_1MA, 
                    u, e, var, h, G, T_arr)   
  return AMA_1,AMA_1AR,GAR_1,GAR_1MA, e, var, h



def round(arr, panel):
  #There may be small differences in calculation between different systems. 
  #For consistency, the inverted matrixes are slightly rounded
  n_digits = panel.options.ARMA_round.value
  zeros = arr==0
  arrz = arr + zeros
  s = np.sign(arr)
  digits = np.array(np.log10(np.abs(arrz)), dtype=int)-(arr<1)
  pow = (n_digits-digits)
  #items smaller in magnitude than e-300 are set to zero:
  arrz = arrz*(pow<300)
  pow = pow*(pow<300)
  a = np.array(arrz*10.0**(pow)+s*0.5,dtype=np.int64)
  a = a*(zeros==False) 
  return a*10.0**(-pow)




