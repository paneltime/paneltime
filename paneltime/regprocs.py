#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import statproc as stat
import functions as fu


def dd_func_lags_mult(panel,ll,AMAL,de_xi,de_zeta,vname1,vname2,transpose=False, de_zeta_u=None):
	#de_xi is "N x T x m", de_zeta is "N x T x k" and L is "T x T"
	
	if de_xi is None or de_zeta is None:
		return None,None	
	(N,T,m)=de_xi.shape
	(N,T,k)=de_zeta.shape
	DLL_e=ll.DLL_e.reshape(N,T,1,1)
	u_calc=False
	if de_zeta_u is None:
		de_zeta_u=de_zeta#for error beta-rho covariance, the u derivative must be used
	#ARIMA:
	if not AMAL is None:
		de2_zeta_xi=fu.dot(AMAL,de_zeta_u,False)#"T x N x s x m"
		if transpose:#only happens if lags==k
			de2_zeta_xi=de2_zeta_xi+np.swapaxes(de2_zeta_xi,2,3)#adds the transpose
		de2_zeta_xi_RE=ddRE(ll,panel,de2_zeta_xi,de_xi,de_zeta,ll.e,vname1,vname2)
	else:
		de2_zeta_xi=0
		de2_zeta_xi_RE=ddRE(ll,panel,None,de_xi,de_zeta,ll.e,vname1,vname2)
		if de2_zeta_xi_RE is None:
			de2_zeta_xi_RE=None
	if not de2_zeta_xi_RE is None:	
		de2_zeta_xi_RE = de2_zeta_xi_RE * DLL_e
		de2_zeta_xi_RE = np.sum(np.sum(de2_zeta_xi_RE,0),0)#and sum it

	#GARCH: 
	if panel.m>0:
		h_e_de2_zeta_xi = de2_zeta_xi * ll.h_e_val.reshape(N,T,1,1)
		h_2e_dezeta_dexi = ll.h_2e_val.reshape(N,T,1,1) * de_xi.reshape((N,T,m,1)) * de_zeta.reshape((N,T,1,k))

		d2lnv_zeta_xi = (h_e_de2_zeta_xi + h_2e_dezeta_dexi)
		
		d_mu = ll.args_d['mu'] * (np.sum(d2lnv_zeta_xi,1) / panel.T_arr.reshape((N,1,1)))
		d_mu = d_mu.reshape((N,1,m,k)) * panel.included.reshape((N,T,1,1))	
		
		
		d2lnv_zeta_xi = fu.dot(ll.GAR_1MA, d2lnv_zeta_xi)
		
		d2lnv_zeta_xi = d2lnv_zeta_xi + d_mu
		
		d2lnv_zeta_xi=np.sum(np.sum(d2lnv_zeta_xi*ll.dLL_lnv.reshape((N,T,1,1)),0),0)
	else:
		d2lnv_zeta_xi=None

	return d2lnv_zeta_xi,de2_zeta_xi_RE

def dd_func_lags(panel,ll,L,d,dLL,addavg=0, transpose=False):
	#d is "N x T x m" and L is "k x T x T"
	if panel.m==0:
		return None
	if d is None:
		return None		
	(N,T,m)=d.shape
	if L is None:
		x=0
	elif len(L)==0:
		return None
	elif len(L.shape)==3:
		x=fu.dot(L,d,False)#"T x N x k x m"
		if transpose:#only happens if lags==k
			x=x+np.swapaxes(x,2,3)#adds the transpose
	elif len(L.shape)==2:
		x=fu.dot(L,d).reshape(N,T,1,m)
	if addavg:
		addavg=(addavg*np.sum(d,1)/panel.T_arr).reshape(N,1,1,m)
		x=x+addavg
	dLL=dLL.reshape((N,T,1,1))
	return np.sum(np.sum(dLL*x,1),0)#and sum it	



def RE(ll,panel,e,recalc=True):
	"""Following Greene(2012) p. 413-414"""
	if panel.FE_RE==0:
		return e
	ll.eFE=FE(panel,e)
	if panel.FE_RE==1:
		return ll.eFE
	if recalc:
		ll.vLSDV=np.sum(ll.eFE**2)/panel.NT_afterloss
		ll.theta=(1-np.sqrt(ll.vLSDV/(panel.grp_v*panel.n_i)))*panel.included
		ll.theta=np.maximum(ll.theta,0)
		if np.any(ll.theta>1):
			raise RuntimeError("WTF")
	eRE=FE(panel,e,ll.theta)
	return eRE

def dRE(ll,panel,de,e,vname):
	"""Returns the first and second derivative of RE"""
	if not hasattr(ll,'deFE'):
		ll.deFE=dict()
		ll.dvLSDV=dict()
		ll.dtheta=dict()

	if panel.FE_RE==0:
		return de
	elif panel.FE_RE==1:
		return FE(panel,de)	
	sqrt_expr=np.sqrt(1/(panel.grp_v*panel.n_i*ll.vLSDV))
	ll.deFE[vname]=FE(panel,de)
	ll.dvLSDV[vname]=np.sum(np.sum(ll.eFE*ll.deFE[vname],0),0)/panel.NT_afterloss
	ll.dtheta[vname]=-sqrt_expr*ll.dvLSDV[vname]*panel.included

	dRE0=FE(panel,de,ll.theta)
	dRE1=FE(panel,e,ll.dtheta[vname],True)
	return (dRE0+dRE1)*panel.included

def ddRE(ll,panel,dde,de1,de2,e,vname1,vname2):
	"""Returns the first and second derivative of RE"""
	if panel.FE_RE==0:
		return dde
	elif panel.FE_RE==1:
		return FE(panel,dde)	
	(N,T,k)=de1.shape
	(N,T,m)=de2.shape
	if dde is None:
		ddeFE=0
		hasdd=False
	else:
		ddeFE=FE(panel,dde)
		hasdd=True
	eFE=ll.eFE.reshape(N,T,1,1)
	ddvLSDV=np.sum(np.sum(eFE*ddeFE+ll.deFE[vname1].reshape(N,T,k,1)*ll.deFE[vname2].reshape(N,T,1,m),0),0)/panel.NT_afterloss

	ddtheta1=(np.sqrt(1/(panel.grp_v*panel.n_i*(ll.vLSDV**2))))*ll.dvLSDV[vname1].reshape(k,1)*ll.dvLSDV[vname2].reshape(1,m)
	ddtheta2=ddtheta1+(-np.sqrt(1/(panel.grp_v*panel.n_i*ll.vLSDV)))*ddvLSDV
	ddtheta=ddtheta2.reshape(N,1,k,m)*panel.included.reshape(N,T,1,1)

	if hasdd:
		dRE00=FE(panel,dde,ll.theta.reshape(N,T,1,1))
	else:
		dRE00=0
	dRE01=FE(panel,de1.reshape(N,T,k,1),ll.dtheta[vname2].reshape(N,T,1,m),True)
	dRE10=FE(panel,de2.reshape(N,T,1,m),ll.dtheta[vname1].reshape(N,T,k,1),True)
	dRE11=FE(panel,e.reshape(N,T,1,1),ddtheta,True)
	return (dRE00+dRE01+dRE10+dRE11)

def FE(panel,e,w=1,d=False):
	"""returns x after fixed effects, and set lost observations to zero"""
	#assumes e is a "N x T x k" matrix
	if e is None:
		return None
	if len(e.shape)==3:
		N,T,k=e.shape
		s=((N,1,k),(N,T,1))
		n_i=panel.n_i
	elif len(e.shape)==4:
		N,T,k,m=e.shape
		s=((N,1,k,m),(N,T,1,1))
		n_i=panel.n_i.reshape((N,1,1,1))
	ec=e*panel.included.reshape(s[1])
	sum_ec=np.sum(ec,1).reshape(s[0])
	sum_ec_all=np.sum(sum_ec,0)	
	dFE=-(w*(sum_ec/n_i-sum_ec_all/panel.NT_afterloss))*panel.included.reshape(s[1])
	if d==False:
		return ec*panel.included.reshape(s[1])+dFE
	else:
		return dFE


def add(iterable,ignore=False):
	"""Sums iterable. If ignore=True all elements except those that are None are added. If ignore=False, None is returned if any element is None. """
	x=None
	for i in iterable:
		if not i is None:
			if x is None:
				x=i
			else:
				x=x+i
		else:
			if not ignore:
				return None
	return x

def prod(iterable,ignore=False):
	"""Takes the product sum of iterable. If ignore=True all elements except those that are None are multiplied. 
	If ignore=False, None is returned if any element is None. """
	x=None
	for i in iterable:
		if not i is None:
			if x is None:
				x=i
			else:
				x=x*i
		else:
			if not ignore:
				return None
	return x

def fillmatr(X,max_T):
	k=len(X[0])
	z=np.zeros((max_T-len(X),k))
	X=np.concatenate((X,z),0)
	return X

def concat_matrix(block_matrix):
	m=[]
	for i in range(len(block_matrix)):
		r=block_matrix[i]
		C=[]
		for j in range(len(r)):
			if not r[j] is None:
				C.append(r[j])
		if len(C):
			m.append(np.concatenate(C,1))
	m=np.concatenate(m,0)
	return m

def concat_marray(matrix_array):
	arr=[]
	for i in matrix_array:
		if not i is None:
			arr.append(i)
	arr=np.concatenate(arr,2)
	return arr

h_err=""

def redefine_h_func(h_definition):
	global h
	if h_definition is None:
		h_definition="""
def h(e,z):
	e2=e**2+z
	i=np.abs(e2)<1e-100
	h_val		=	 np.log(e2+i)	
	h_e_val		=	 2*e/(e2+i)
	h_2e_val	=	 2*(z-e**2)/((e2+i)**2)
	h_z_val		=	 1/(e2+i)
	h_2z_val	=	-1/(e2+i)**2
	h_ez_val	=	-2*e/(e2+i)**2
	return h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val
"""	
	d=dict()
	try:
		exec(h_definition,globals(),locals())
	except IndentationError:
		h_list=h_definition.split('\n')
		n=h_list[0].find('def h(')
		if n<0:
			raise RuntimeError('The h-funtion must be defined as  "def h(..."')
		if n>0:
			for i in range(len(h_list)):
				h_list[i]=h_list[i][n:]
		h_definition='\n'.join(h_list)
		exec(h_definition,globals(),locals())
		pass
	h=locals()['h']

def h_func(e,z):
	global h_err
	try:
		return h(e, z)
	except Exception as err:
		if h_err!=str(err):
			print ("Warning: error in the ARCH error function h(e,z). The error was: %s" %(err))
		h_err=str(e)
	else:
		h_err="none"

def format_args_array(arg_array,master):
	for i in range(len(arg_array)):
		arg_array[i]=format_args(arg_array[i], master)
	return arg_array
		
def format_args(x,master):
	if master is None:
		x=x.replace('rp.','')
	x=x.replace('\t','')
	while 1:
		x=x.replace('  ',' ')
		x=x.replace('\n ','\n')
		x=x.replace('\n\n','\n')
		if not (('  ' in x) or ('\n ' in x) or ('\n\n' in x)):
			break
	if x[0]=='\n': x=x[1:]
	if x[len(x)-1]=='\n': x=x[:len(x)-1] 
	return x
			
class multiprocess:
	def __init__(self,hessian):
		self.master=hessian.master
		self.d=dict()
		if not self.master is None:
			self.master.send_holdbacks(['AMAp','AMAq','GARM','GARK'])
		
	def execute(self,expr,master):
		"""For submitting multiple functionsargs is an array of argument arrays where the first element in each 
		argument array is the function to be evaluated"""
		if master is None:#For debugging purposes
			for i in expr:
				exec(i,None,self.d)#the first element in i is the function, the rest are arguments
		else:
			d=self.master.send_tasks(expr)
			for i in d:
				self.d[i]=d[i]			
		return self.d
	
		
	def send_dict(self,d,instructions):
		for i in d:
			self.d[i]=d[i]
		if str(type(self.master))=="<class 'multi_core.master'>":
			self.master.send_dict(d,instructions)
	
def make_task_dict(expr,task_types):
	d=dict()
	d['all']=[]#generic task
	for e in expr:
		t_found=None
		for t in task_types:
			if t in e:
				t_found=t
				break
		if not t_found is None:
			if not t_found in d:
				d[t_found]=[]
			d[t_found].append(e)
		else:
			d['all'].append(e)
	return d
		
def dd_func(d2LL_de2,d2LL_dln_de,d2LL_dln2,de_dh,de_dg,dln_dh,dln_dg,dLL_de2_dh_dg,dLL_dln2_dh_dg):
	a=[]
	a.append(dd_func_mult(de_dh,d2LL_de2,de_dg))

	a.append(dd_func_mult(de_dh,d2LL_dln_de,dln_dg))
	a.append(dd_func_mult(dln_dh,d2LL_dln_de,de_dg))

	a.append(dd_func_mult(dln_dh,d2LL_dln2,dln_dg))

	a.append(dLL_de2_dh_dg)
	a.append(dLL_dln2_dh_dg)
	return add(a,True)

def dd_func_mult(d0,mult,d1):
	#d0 is N x T x k and d1 is N x T x m
	if d0 is None or d1 is None or mult is None:
		return None
	(N,T,k)=d0.shape
	(N,T,m)=d1.shape
	d0=d0*mult
	d0=np.reshape(d0,(N,T,k,1))
	d1=np.reshape(d1,(N,T,1,m))
	x=np.sum(np.sum(d0*d1,0),0)#->k x m 
	return x


def ARMA_product(m,L,k):
	a=[]
	for i in range(k):
		a.append(fu.dot(m,L[i]))
	return np.array(a)

			
def solve(constr,H, g, x):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	n=len(H)
	c,c_eq=constr.constraints_to_arrays()
	k=len(c)
	m=len(c_eq)
	H=np.concatenate((H,np.zeros((n,k+m))),1)
	H=np.concatenate((H,np.zeros((k+m,n+k+m))),0)
	g=np.append(g,(k+m)*[0])
	
	
	r_eq_indicies=[]
	for i in range(k+m):
		H[n+i,n+i]=1
	for i in range(m):
		j=int(c_eq[i][1])
		H[j,n+i]=1
		H[n+i,j]=1
		H[n+i,n+i]=0
		g[n+i]=-(c_eq[i][0]-x[j])
		r_eq_indicies.append(j)

	xi=-np.linalg.solve(H,g).flatten()
	for i in range(k):#Kuhn-Tucker:
		j=int(c[i][2])
		q=None
		if j in r_eq_indicies:
			q=None
		elif x[j]+xi[j]<c[i][0]-1e-15:
			q=-(c[i][0]-x[j])
		elif x[j]+xi[j]>c[i][1]+1e-15:
			q=-(c[i][1]-x[j])
		if q!=None:
			H[j,n+i+m]=1
			H[n+i+m,j]=1
			H[n+i+m,n+i+m]=0
			g[n+i+m]=q
			xi=-np.linalg.solve(H,g).flatten()	
	constrained=np.sum(H[n:,:n],0)
	return xi[:n],constrained

def sandwich(H,G,lags=3):
	hessin=np.linalg.inv(-H)
	V=stat.newey_west_wghts(lags,XErr=G)
	hessinV=fu.dot(hessin,V)
	sandwich=fu.dot(hessinV,hessin)
	return sandwich

def add_names(T,namsestr,names,start=0):
	a=[]
	for i in range(start,T):
		a.append(namsestr+str(i))
	names.extend(a)
	
def remove_constants(panel,G,include,constr,out,names):
	N,T,k=G.shape
	v=stat.var(panel,G)
	for i in range(1,k):
		if v[0][i]==0:
			include[i]=False
			constr.add(i,0)
			out.add(names[i],0,'NA','constant')	
			
			
def remove_H_correl(panel,hessian,include,constr,args,out,names):
	k,k=hessian.shape
	hessian_abs=np.abs(hessian)
	x=(np.diag(hessian_abs)**0.5).reshape((1,k))
	x=(x.T*x)
	corr=hessian_abs/(x+(x==0)*1e-100)	
	for i in range(k):
		m=np.max(corr[i])
		if m>2*corr[i,i]:
			j=np.nonzero(corr[i]==m)[0][0]
			corr[:,j]=0
			corr[j,:]=0
			corr[j,j]=1	
	for i in range(k):
		corr[i,i:]=0

	p=np.arange(k).reshape((1,k))*np.ones((k,1))
	p=np.concatenate((corr.reshape((k,k,1)),p.T.reshape((k,k,1)),p.reshape((k,k,1))),2)
	p=p.reshape((k*k,3))
	srt=np.argsort(p[:,0],0)
	p=p[srt][::-1]
	p=p[np.nonzero(p[:,0]>=1.0)[0]]
	principal_factors=[]
	groups=correl_groups(p)
	acc=None
	for i in groups:
		for j in range(len(i)):
			if not i[j] in constr.constraints:
				acc=i.pop(j)
				break
		if not acc is None:
			for j in i:
				remvd=remove(j,acc,args,include,out,constr,names,'h-correl')	
	return hessian

def remove_correl(panel,G,include,constr,args,out,names):
	N,T,k=G.shape
	corr=np.abs(stat.correl(G,panel))
	for i in range(k):
		corr[i,i:]=0

	p=np.arange(k).reshape((1,k))*np.ones((k,1))
	p=np.concatenate((corr.reshape((k,k,1)),p.T.reshape((k,k,1)),p.reshape((k,k,1))),2)
	p=p.reshape((k*k,3))
	srt=np.argsort(p[:,0],0)
	p=p[srt][::-1]
	p=p[np.nonzero(p[:,0]>0.8)[0]]
	principal_factors=[]
	groups=correl_groups(p)
	for i in groups:
		for j in range(len(i)):
			if not i[j] in constr.constraints:
				acc=i.pop(j)
				break
		for j in i:
			remvd=remove(j,acc,args,include,out,constr,names,'correl')	


def append_to_group(group,intlist):
	ingroup=False
	for i in intlist:
		if i in group:
			ingroup=True
			break
	if ingroup:
		for j in intlist:
			if not j in group:
				group.append(j)
		return True
	else:
		return False
	
def correl_groups(p):
	groups=[]
	appended=False
	x=np.array(p[:,1:3],dtype=int)
	for i,j in x:
		for k in range(len(groups)):
			appended=append_to_group(groups[k],[i,j])
			if appended:
				break
		if not appended:
			groups.append([i,j])
	g=len(groups)
	keep=g*[True]
	for k in range(g):
		if keep[k]:
			for h in range(k+1,len(groups)):
				appended=False
				for m in range(len(groups[h])):
					if groups[h][m] in groups[k]:
						appended=append_to_group(groups[k],  groups[h])
						keep[h]=False
						break
	g=[]
	for i in range(len(groups)):
		if keep[i]:
			g.append(groups[i])
	return g
		
	
def remove_one_multicoll(G,panel,args,names,include,out,constr,limit):
	n=len(include)
	T,N,k=G.shape
	c_index,var_prop=stat.var_decomposition(X=G[:,:,include])
	zeros=np.zeros(len(c_index))
	c_index=c_index.flatten()
	for i in range(k):
		if not include[i]:
			c_index=np.insert(c_index,i,0)
			var_prop=np.insert(var_prop,i,zeros,1)
	
	if c_index[-1]>limit:
		if np.sum(var_prop[-1]>0.5)>1:
			j=np.argsort(var_prop[-1])[-1]
			assc=np.argsort(var_prop[-1])[-2]
			remvd=remove(j, assc,args, include, out,constr,names,'collinear')
			return True
	return False

def remove_all_multicoll(G,panel,args,names,include,out,constr,limit):
	T,N,k=G.shape
	for i in range(k):
		remvd=remove_one_multicoll(G,panel,args,names,include,out,constr,limit)
		if not remvd:
			return


def remove(d,assoc,args,include,out,constr,names,r_type):
	if d in constr.constraints:
		return False
	constr.add(d,args[d])
	include[d]=False
	if not assoc is None:
		out.add(names[d],args[d],names[assoc],r_type)	
	else:
		out.add(names[d],args[d],'NA',r_type)	
	return True

def handle_multicoll(G,panel,args,names,constr,mc_limit,dx_conv,hessian,has_problems,k,its):
	N,T,h=G.shape
	include=np.ones(h,dtype=bool)
	out=output()
	remove_constants(panel, G, include,constr,out,names)	
	remove_all_multicoll(G, panel, args, names, include, out, constr, 300)
	reset=False
	remove_H_correl(panel,hessian,include,constr,args,out,names)
	if mc_limit<30:
		srt=np.argsort(dx_conv)
		for i in range(min((k,len(srt)-2))):
			j=srt[-i-1]
			if dx_conv[j]<0.05:
				reset=True
			else:
				reset=remove(j,None,args, include, out,constr,names,'dir cap')==False
	return hessian, reset,out
		

class output:
	def __init__(self):
		self.variable=[]
		self.set_to=[]
		self.assco=[]
		self.cause=[]

	def add(self,variable,set_to,assco,cause):
		if (not (variable in self.variable)) or (not (cause in self.cause)):
			self.variable.append(variable)
			self.set_to.append(str(round(set_to,8)))
			self.assco.append(assco)
			self.cause.append(cause)

	def print(self):
		output= "|Restricted variable |    Set to    | Associated variable|  Cause   |\n"
		output+="|--------------------|--------------|--------------------|----------|\n"
		if len(self.variable)==0:
			return
		for i in range(len(self.variable)):
			output+="|%s|%s|%s|%s|\n" %(
			    self.variable[i].ljust(20)[:20],
			    self.set_to[i].rjust(14)[:14],
			    self.assco[i].ljust(20)[:20],
			    self.cause[i].ljust(10)[:10])	

		print(output)	

	
def differenciate(X,diff,has_intercept):
	for i in range(diff):
		X=X-np.roll(X,1,0)
	X=X[diff:]
	if has_intercept:
		X[:,0]=1
	return X

		
def make_lag_matrices(T,n):
	L0=np.diag(np.ones(T-1),-1)
	L=[L0]
	for i in range(n-1):
		L.append(fu.dot(L0,L[i]))
	return L


		

				