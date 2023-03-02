#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pydoc import importfile
import os
path = os.path.dirname(__file__)
stat =  importfile(os.path.join(path,'stat_functions.py'))


import numpy as np
import stat_functions as stat


def get(g, x, H, constr, f, hessin, ev_constr, simple=True):
  n = len(x)
  if simple or (H is None):
    dx = -(np.dot(hessin,g.reshape(n,1))).flatten()
  else:
    #dx, H =solve(self.H, self.g, args.args_v, self.constr, f)
    dx, H = solve_delete(constr, H, g, x, f, ev_constr)	
  dx_norm = normalize(dx, x)

  return dx, dx_norm, H



def solve(H, g, x, constr, f, ev_constr):
  """Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
  g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
  and index constrained indicating the constrained variables"""
  if H is None:
    raise RuntimeError('Hessian is None')


  dx_init, H = linalg_solve(H, g, f, x, ev_constr)
  if constr is None:
    return dx_init	
  n=len(H)
  k=len(constr)
  H=np.concatenate((H,np.zeros((n,k))),1)
  H=np.concatenate((H,np.zeros((k,n+k))),0)
  g=np.append(g,(k)*[0])

  for i in range(k):
    H[n+i,n+i]=1
  j=0
  dx=np.zeros(len(g))
  for i in constr.fixed:
    #kuhn_tucker(constr.fixed[i],i,j,n, H, g, x,dx, recalc=False)
    kuhn_tucker2(constr.fixed[i],i,j,n, H, g, x, f,dx, dx_init, ev_constr, recalc=False)
    j+=1
  dx, H = linalg_solve(H, g, f, x, ev_constr)	
  OK=False
  w=0
  for r in range(50):
    j2=j

    for i in constr.intervals:
      dx, H=kuhn_tucker(constr.intervals[i],i,j2,n, H, g, x, f,dx, ev_constr)
      j2+=1
    OK=constr.within(x+dx[:n],False)
    if OK: 
      break
    if r==k+3:
      #print('Unable to set constraints in computation calculation')
      break

  return dx[:n], H


def solve_delete(constr,H, g, x, f, ev_constr):
  """Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
  g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
  and index constrained indicating the constrained variables"""

  if H is None:
    raise RuntimeError('Cant solve with no coefficient matrix')
  try:
    list(constr.keys())[0]
  except:
    dx, H = linalg_solve(H, g, f, x, ev_constr)
    return dx

  H_orig = np.array(H)
  m=len(H)

  idx=np.ones(m,dtype=bool)
  delmap=np.arange(m)
  if len(list(constr.fixed.keys()))>0:#removing fixed constraints from the matrix
    idx[list(constr.fixed.keys())]=False
    H=H[idx][:,idx]
    g=g[idx]
    delmap-=np.cumsum(idx==False)
    delmap[idx==False]=m#if for some odd reason, the deleted variables are referenced later, an out-of-bounds error is thrown
  n=len(H)
  k=len(constr.intervals)
  H=np.concatenate((H,np.zeros((n,k))),1)
  H=np.concatenate((H,np.zeros((k,n+k))),0)
  g=np.append(g,(k)*[0])

  for i in range(k):
    H[n+i,n+i]=1

  dx, H = linalg_solve(H, g, f, x, ev_constr)
  xi_full=np.zeros(m)
  OK=False
  keys=list(constr.intervals.keys())
  for r in range(50):		
    for j in range(k):
      key=keys[j]
      dx, H =kuhn_tucker_del(constr,key,j,n, H, g, x, f,dx,delmap, ev_constr)
    xi_full[idx]=dx[:n]
    OK=constr.within(x+xi_full,False)
    if OK: 
      break
    if r==k+3:
      #print('Unable to set constraints in computation calculation')
      break
  xi_full=np.zeros(m)
  xi_full[idx]=dx[:n]
  H_full = np.zeros((m,m))
  idx = idx.reshape((1,m))
  nz = np.nonzero(idx*idx.T)
  H_full[nz] = H[np.nonzero(np.ones((n,n)))]
  H_full[H_full==0] = H_orig[H_full==0]
  return xi_full, H_full


def kuhn_tucker_del(constr,key,j,n,H,g,x, f, dx,delmap, ev_constr,recalc=True):
  q=None
  c=constr.intervals[key]
  i=delmap[key]
  if not c.value is None:
    q=-(c.value-x[i])
  elif x[i]+dx[i]<c.min:
    q=-(c.min-x[i])
  elif x[i]+dx[i]>c.max:
    q=-(c.max-x[i])
  if q!=None:
    H[i,n+j]=1
    H[n+j,i]=1
    H[n+j,n+j]=0
    g[n+j]=q
    if recalc:
      dx, H = linalg_solve(H, g, f, x, ev_constr)
  return dx, H


def kuhn_tucker(c,i,j,n,H,g,x,f,dx, ev_constr,recalc=True):
  q=None
  if not c.value is None:
    q=-(c.value-x[i])
  elif x[i]+dx[i]<c.min:
    q=-(c.min-x[i])
  elif x[i]+dx[i]>c.max:
    q=-(c.max-x[i])
  if q!=None:
    H[i,n+j]=1
    H[n+j,i]=1
    H[n+j,n+j]=0
    g[n+j]=q
    if recalc:
      dx, H = linalg_solve(H, g, f, x, ev_constr)
  return dx, H


def kuhn_tucker2(c,i,j,n,H,g,x, f,dx,dx_init, ev_constr,recalc=True):
  if c.assco_ix is None:
    return kuhn_tucker(c,i,j,n,H,g,x, f, dx,ev_constr, recalc)
  q=None
  if not c.value is None:
    q=-(c.value-x[i])
  elif x[i]+dx[i]<c.min:
    q=-(c.min-x[i])
  elif x[i]+dx[i]>c.max:
    q=-(c.max-x[i])
  if q!=None:
    H[i,n+j]=1
    H[n+j,i]=1
    H[n+j,n+j]=0
    g[n+j]=q
    if recalc:
      dx, H = linalg_solve(H, g, f, x, ev_constr)
  return dx, H




def normalize(dx, x):
  x = np.abs(x)
  dx_norm=(x!=0)*dx/(x+(x<1e-100))
  dx_norm=(x<1e-2)*dx+(x>=1e-2)*dx_norm	
  return dx_norm	



def ev_non_sing(H, ev_constr):
  c, var_prop, ev, p = stat.var_decomposition(XXNorm=H)
  if not ev_constr:
    return ev,p
  c = c.flatten()
  p = p.real
  limit = 100
  v = np.sum(var_prop>0.5,1)<2
  keep = (v|(c<limit))
  ev = ev * keep
  return ev, p

def linalg_solve(H,g, f, x, ev_constr):

  ev, p = ev_non_sing(H, ev_constr)
  a = np.dot(p, p.T)
  b = np.dot(p.T, p)

  gp = np.dot(g, p)

  negev = (ev<0)+(ev==1.0)
  ev = ev*negev - ev*(negev==False)
  ev_1 = 1/(ev + (ev==0))

  gpL = gp*ev_1*(ev!=0)

  dx = -np.real(np.dot(gpL, p.T))

  H = np.dot(p, np.dot(np.diag(ev), p.T))


  return dx, H

def linalg_solve_old(H,g):
  try:
    return -np.linalg.solve(H,g).flatten(), H
  except np.linalg.LinAlgError as e:
    pass
  Hd = np.diag(H)
  Hd = Hd*(np.abs(Hd)>1e-30)
  Hd[Hd==0] = np.min(Hd[Hd<0])
  Hd = np.diag(Hd)
  H = H + Hd
  try:
    return -np.linalg.solve(H,g).flatten(), H
  except np.linalg.LinAlgError as e:
    pass		

  H = H - np.abs(Hd)
  try:
    return -np.linalg.solve(H,g).flatten(), H
  except np.linalg.LinAlgError as e:
    pass

  H =  - np.abs(Hd)
  try:
    return -np.linalg.solve(H,g).flatten(), H
  except np.linalg.LinAlgError as e:
    raise RuntimeError('Unable to fix singularity')		

  H = np.identity(len(Hd))
  return -np.linalg.solve(H,g).flatten(), H
