#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object
import sys
#sys.path.append(__file__.replace("paneltime\\loglikelihood.py",'build\\lib.win-amd64-3.5'))
#sys.path.append(__file__.replace("paneltime\\loglikelihood.py",'build\\lib.linux-x86_64-3.5'))
try:#only using c function if installed
	import cfunctions as c
except ImportError as e:
	c=None
import numpy as np
import functions as fu
import regprocs as rp
import statproc as stat
import random_effects as re
import calculus
from scipy import sparse as sp
import scipy

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and creates an object
	that store dynamic variables that depend on the \n
	If args is a dictionary, the ARMA-GARCH orders are 
	determined from the dictionary. If args is a vector, the ARMA-GARCH order needs to be consistent
	with the  panel object
	"""
	def __init__(self,args,panel,X=None):
		
		self.panel=panel
		self.re_obj=re.re_obj(panel)
		if args is None:
			args=panel.args.args
		self.LL_const=-0.5*np.log(2*np.pi)*panel.NT
	
		self.args_v=panel.args.conv_to_vector(args)
		self.args_d=panel.args.conv_to_dict(args)
		self.h_err=""
		self.h_def=panel.h_def
		
		
		try:
			self.LL=self.LL_calc(panel,X)
			
		except Exception as e:
			self.LL=None
			print(str(e))
		if not self.LL is None:
			if np.isnan(self.LL):
				self.LL=None
		
		


	def LL_calc(self,panel,X=None):
		args=self.args_d#using dictionary arguments
		if X is None:
			X=panel.X
		matrices=set_garch_arch(panel,args)
		if matrices is None:
			return None		
		
		AMA_1,AMA_1AR,GAR_1,GAR_1MA=matrices
		(N,T,k)=panel.X.shape

		u=panel.Y-fu.dot(panel.X,args['beta'])
		e=fu.dot(AMA_1AR,u)
		
		if panel.m>0:
			h_res=self.h(e, args['z'][0])
			if h_res==None:
				return None
			(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val)=[i*panel.included for i in h_res]
			lnv_ARMA=fu.dot(GAR_1MA,h_val)
		else:
			(h_val,h_e_val,h_2e_val,h_z_val,h_2z_val,h_ez_val,avg_h)=(0,0,0,0,0,0,0)
			lnv_ARMA=0	
		
		W_omega=fu.dot(panel.W_a,args['omega'])
		lnv=W_omega+lnv_ARMA# 'N x T x k' * 'k x 1' -> 'N x T x 1'
		if panel.m>0:
			if False: #using average of h function as group variance
				avg_h=panel.mean(h_val,1).reshape((N,1,1))*panel.a
				self.avg_lne2,self.davg_lne2,self.d2avg_lne2   = avg_h,     h_e_val,   h_2e_val
				self.avg_e2, self.zmu                          = None,  1
				
			else: #using log average of e**2 function as group variance
				avg_e2=panel.mean(e**2,1).reshape((N,1,1))
				avg_lne2=np.log(avg_e2)*panel.a
				#the derivatives, for later use
				avg_e=panel.mean(e,1).reshape((N,1,1))
				davg_lne2=(2*e/avg_e2)
				d2avg_lne2=(2/avg_e2-4*(e*avg_e/avg_e2**2))		
			
				self.avg_lne2,self.davg_lne2,self.d2avg_lne2   = avg_lne2,  davg_lne2, d2avg_lne2 
				self.avg_e2, self.zmu                          = avg_e2, 0			
			if panel.N>1:
				lnv=lnv+args['mu'][0]*self.avg_lne2
			lnv=np.maximum(np.minimum(lnv,100),-100)
		v=np.exp(lnv)*panel.a
		v_inv=np.exp(-lnv)*panel.a	
		e_RE=self.re_obj.RE(e)
		e_REsq=e_RE**2
		LL=self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
		
		if abs(LL)>1e+100: 
			return None
		self.AMA_1,self.AMA_1AR,self.GAR_1,self.GAR_1MA=matrices
		self.u,self.e,self.h_val, self.lnv_ARMA        = u,         e,         h_val,    lnv_ARMA
		self.lnv,self.v,self.v_inv                     = lnv,       v,         v_inv
		self.e_RE,self.e_REsq                          = e_RE,      e_REsq
		self.h_z_val,self.h_ez_val,self.h_2z_val       = h_z_val,   h_ez_val,  h_2z_val
		self.h_e_val,  self.h_2e_val                   = h_e_val,   h_2e_val

		
		return LL
	

	def standardize(self):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self"""
		sd_inv=self.v_inv**0.5
		panel=self.panel
		m=panel.lost_obs
		N,T,k=panel.X.shape
		Y=fu.dot(self.AMA_1AR,panel.Y)
		Y=self.re_obj.RE(Y,False)*sd_inv
		X=fu.dot(self.AMA_1AR,panel.X)
		X=self.re_obj.RE(X,False)*sd_inv
		self.e_st=self.e_RE*sd_inv
		self.Y_st=Y
		self.X_st=X
		incl=panel.included.reshape(N,T)
		self.e_st_long=self.e_st[incl,:]
		self.Y_st_long=self.Y_st[incl,:]
		self.X_st_long=self.X_st[incl,:]

	def copy_args_d(self):
		return fu.copy_array_dict(self.args_d)

	
	def h(self,e,z):
		d={'e':e,'z':z}
		try:
			exec(self.h_def,globals(),d)
		except Exception as err:
			if self.h_err!=str(err):
				print ("Warning,error in the ARCH error function h(e,z): %s" %(err))
			h_err=str(e)
			return None
	
		return d['ret']	

def set_garch_arch(panel,args):
	if c is None:
		m=set_garch_arch_scipy(panel,args)
	else:
		m=set_garch_arch_c(panel,args)
	return m
		
		
def set_garch_arch_c(panel,args):
	"""Solves X*a=b for a where X is a banded matrix with 1 or zero, and args along
	the diagonal band"""
	n=panel.max_T
	rho=np.insert(-args['rho'],0,1)
	psi=np.insert(args['psi'],0,0)

	r=np.arange(n)
	AMA_1,AMA_1AR,GAR_1,GAR_1MA=(
	    np.diag(np.ones(n)),
		np.zeros((n,n)),
		np.diag(np.ones(n)),
		np.zeros((n,n))
	)
	c.bandinverse(args['lambda'],rho,-args['gamma'],psi,n,AMA_1,AMA_1AR,GAR_1,GAR_1MA)
	return  AMA_1,AMA_1AR,GAR_1,GAR_1MA


def set_garch_arch_scipy(panel,args):

	p,q,m,k,nW,n=panel.p,panel.q,panel.m,panel.k,panel.nW,panel.max_T

	AAR=-lag_matr(-panel.I,args['rho'])
	AMA_1AR,AMA_1=solve_mult(args['lambda'], AAR, panel.I)
	if AMA_1AR is None:
		return
	GMA=lag_matr(panel.zero,args['psi'])	
	GAR_1MA,GAR_1=solve_mult(-args['gamma'], GMA, panel.I)
	if GAR_1MA is None:
		return
	return AMA_1,AMA_1AR,GAR_1,GAR_1MA
	
def solve_mult(args,b,I):
	"""Solves X*a=b for a where X is a banded matrix with 1  and args along
	the diagonal band"""
	n=len(b)
	q=len(args)
	X=np.zeros((q+1,n))
	X[0,:]=1
	X2=np.zeros((n,n))
	w=np.zeros(n)
	r=np.arange(n)	
	for i in range(q):
		X[i+1,:n-i-1]=args[i]
	try:
		X_1=scipy.linalg.solve_banded((q,0), X, I)
		if np.any(np.isnan(X_1)):
			return None,None			
		X_1b=fu.dot(X_1, b)
	except:
		return None,None

	return X_1b,X_1


def add_to_matrices(X_1,X_1b,a,ab,r):
	for i in range(0,len(a)):	
		if i>0:
			d=(r[i:],r[:-i])
			X_1[d]=a[i]
		else:
			d=(r,r)
		X_1b[d]=ab[i]	
	return X_1,X_1b

def lag_matr(L,args):
	k=len(args)
	if k==0:
		return L
	L=1*L
	r=np.arange(len(L))
	for i in range(k):
		d=(r[i+1:],r[:-i-1])
		L[d]=args[i]

	return L


class direction:
	def __init__(self,panel):
		self.gradient=calculus.gradient(panel)
		self.hessian=calculus.hessian(panel,self.gradient)
		self.panel=panel
		self.constr=None
		self.hessian_num=None
		self.g_old=None
		self.do_shocks=True
		self.old_dx_conv=None
		self.I=np.diag(np.ones(panel.args.n_args))
		
		
	def get(self,ll,mc_limit,dx_conv,k,its,mp=None,dxi=None,user_constraints=None,numerical=False):

		g,G=self.gradient.get(ll,return_G=True)		
		hessian=self.get_hessian(ll,mp,g,G,dxi,its,dx_conv,numerical)

		out=output()
		self.constr=constraints(self.panel.args,self.constr)
		reset=False
		hessian,reset=add_constraints(G,self.panel,ll,self.constr,mc_limit,dx_conv,self.old_dx_conv,hessian,k,its,out,user_constraints)
		self.old_dx_conv=dx_conv
		dc,constrained=solve(self.constr,hessian, g, ll.args_v)
		for j in range(len(dc)):
			s=dc*(constrained==0)*g
			if np.sum(s)<0:#negative slope
				s=np.argsort(s)
				k=s[0]
				remove(k, None, ll.args_v, None, out, self.constr, self.panel.args.names_v, 'neg. slope')
				dc,constrained=solve(self.constr,hessian, g, ll.args_v)
			else:
				break
		
		out.print()
		
			
		return dc,g,G,hessian,constrained,reset
	

	
	def get_hessian(self,ll,mp,g,G,dxi,its,dx_conv,numerical):
		
		#hessinS0=rp.sandwich(hessian,G,0)
		hessian=None
		I=self.I
		if not numerical or self.hessian_num is None:
			hessian=self.hessian.get(ll,mp)
		hessian,num=self.nummerical_hessian(hessian, dxi,g)
		if num:
			return hessian
		if dx_conv is None:
			m=10
		else:
			dx_max=max(dx_conv)
			fact=200
			m=fact*dx_max/(fact*(1/dx_max)+dx_max)
		#m=1
		#if dx_max<0.2:
		#	m=0
				
		hessian=(hessian+m*I*hessian)/(1+m)
		self.hessian_num=hessian
		self.g_old=g
		
		return hessian
	
	def nummerical_hessian(self,hessian,dxi,g):
		if not hessian is None:
			return hessian,False
		I=self.I
		if (self.g_old is None) or (dxi is None):
			return I,True

		#print("Using numerical hessian")		
		hessin_num=hessin(self.hessian_num)
		if hessin_num is None:
			return I,True
		hessin_num=nummerical_hessin(g,self.g_old,hessin_num,dxi)	
		hessian=hessin(hessin_num)
		if hessian is None:
			return I,True
		return hessian,True
		
def hessin(hessian):
	try:
		h=-np.linalg.inv(hessian)
	except:
		return None	
	return h
	
def nummerical_hessin(g,g_old,hessin,dxi):
	if dxi is None:
		return None
	dg=g-g_old 				#Compute difference of gradients,
	#and difference times current matrix:
	n=len(g)
	hdg=(np.dot(hessin,dg.reshape(n,1))).flatten()
	fac=fae=sumdg=sumxi=0.0 							#Calculate dot products for the denominators. 
	fac = np.sum(dg*dxi) 
	fae = np.sum(dg*hdg)
	sumdg = np.sum(dg*dg) 
	sumxi = np.sum(dxi*dxi) 
	if (fac > (3.0e-16*sumdg*sumxi)**0.5):#Skip update if fac not sufficiently positive.
		fac=1.0/fac
		fad=1.0/fae 
								#The vector that makes BFGS different from DFP:
		dg=fac*dxi-fad*hdg   
		#The BFGS updating formula:
		hessin+=fac*dxi.reshape(n,1)*dxi.reshape(1,n)
		hessin-=fad*hdg.reshape(n,1)*hdg.reshape(1,n)
		hessin+=fae*dg.reshape(n,1)*dg.reshape(1,n)	
	return hessin
	
	
def solve(constr,H, g, x):
	"""Solves a second degree taylor expansion for the dc for df/dc=0 if f is quadratic, given gradient
	g, hessian H, inequalty constraints c and equalitiy constraints c_eq and returns the solution and 
	and index constrained indicating the constrained variables"""
	if H is None:
		return None,g*0
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
	sel=[i for i in range(len(H))]
	H[sel,sel]=H[sel,sel]+(H[sel,sel]==0)*1e-15
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

def remove_constants(panel,G,include,constr,out,names):
	N,T,k=G.shape
	v=panel.var(G,(0,1))
	for i in range(1,k):
		if v[i]==0:
			include[i]=False
			constr.set(i,0)
			out.add(names[i],0,'NA','constant')	


def remove_H_correl(hessian,include,constr,args,out,names):
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
	IDs=correl_IDs(p)
	acc=None
	for i in IDs:
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
	IDs=correl_IDs(p)
	for i in IDs:
		for j in range(len(i)):
			if not i[j] in constr.constraints:
				acc=i.pop(j)
				break
		for j in i:
			remvd=remove(j,acc,args,include,out,constr,names,'correl')	


def append_to_ID(ID,intlist):
	inID=False
	for i in intlist:
		if i in ID:
			inID=True
			break
	if inID:
		for j in intlist:
			if not j in ID:
				ID.append(j)
		return True
	else:
		return False

def correl_IDs(p):
	IDs=[]
	appended=False
	x=np.array(p[:,1:3],dtype=int)
	for i,j in x:
		for k in range(len(IDs)):
			appended=append_to_ID(IDs[k],[i,j])
			if appended:
				break
		if not appended:
			IDs.append([i,j])
	g=len(IDs)
	keep=g*[True]
	for k in range(g):
		if keep[k]:
			for h in range(k+1,len(IDs)):
				appended=False
				for m in range(len(IDs[h])):
					if IDs[h][m] in IDs[k]:
						appended=append_to_ID(IDs[k],  IDs[h])
						keep[h]=False
						break
	g=[]
	for i in range(len(IDs)):
		if keep[i]:
			g.append(IDs[i])
	return g


def remove_one_multicoll(G,args,names,include,out,constr,limit):
	n=len(include)
	T,N,k=G.shape
	try:
		c_index,var_prop=stat.var_decomposition(X=G[:,:,include])
	except:
		return False
	zeros=np.zeros(len(c_index))
	c_index=c_index.flatten()
	for i in range(k):
		if not include[i]:
			c_index=np.insert(c_index,i,0)
			var_prop=np.insert(var_prop,i,zeros,1)

	if c_index[-1]>limit:
		if np.sum(var_prop[-1]>0.49)>1:
			j=np.argsort(var_prop[-1])[-1]
			assc=np.argsort(var_prop[-1])[-2]
			remvd=remove(j, assc,args, include, out,constr,names,'collinear')
			return True
	return False

def remove_all_multicoll(G,args,names,include,out,constr,limit):
	T,N,k=G.shape
	for i in range(k):
		remvd=remove_one_multicoll(G,args,names,include,out,constr,limit)
		if not remvd:
			return


def remove(d,assoc,set_to,include,out,constr,names,r_type):
	""""removes" variable d by constraining it to set_to. If an assoc variable is not None, the assoc will be
	printed as an assocaited variable. If set_to is an array, 
	then it is constrained to set_to[d]. include[d] is set to false. out is the output object, constr is 
	the constraints object. name[d] and name[assoc] are printed. the type of removal r_type is also printed."""
	if d in constr.constraints:
		return False

	if type(set_to)==list or type(set_to)==np.ndarray:
		a=set_to[d]
	else:
		a=set_to
	constr.set(d,a,assoc=assoc)
	if not include is None:
		include[d]=False	
	if not assoc is None:
		out.add(names[d],a,names[assoc],r_type)	
	else:
		out.add(names[d],a,'NA',r_type)	
	return True

def add_constraints(G,panel,ll,constr,mc_limit,dx_conv,dx_conv_old,hessian,k,its,out,user_constraints):
	names=panel.args.names_v
	args=ll.args_v
	N,T,h=G.shape
	include=np.ones(h,dtype=bool)
	add_other_constraints(constr,user_constraints,ll)
	general_constraints=[('rho',-2,2),('lambda',-2,2),('gamma',-2,2),('psi',-2,2)]
	add_other_constraints(constr,general_constraints,ll)
	add_initial_constraints(panel,constr,out,names,ll,include,its)
	remove_constants(panel, G, include,constr,out,names)	

	remove_all_multicoll(G, args, names, include, out, constr, 50)
	remove_H_correl(hessian,include,constr,args,out,names)
	reset=False
	if mc_limit<30 and not (dx_conv is None):
		srt=np.argsort(dx_conv)
		for i in range(min((k,len(srt)-2))):
			if max(dx_conv)>2:
				j=srt[-i-1]
			else:
				j=srt[i]
			if dx_conv[j]<0.01:
				reset=True
			else:
				remove(j,None,args, include, out,constr,names,'dir cap')
	return hessian, reset

def add_initial_constraints(panel,constr,out,names,ll,include,its):
	""" "Removes" variables by constraining them to specific values initially """
	args=panel.args
	if its<-3:
		for a in ['beta','rho','gamma','psi','lambda']:
			for i in args.positions[a][1:]:
				remove(i, None, 0, include, out, constr, names, 'initial')
	if its==-3:
		ll.standardize(panel)
		beta=stat.OLS(panel,ll.X_st,ll.Y_st)
		for i in args.positions['beta']:
			remove(i, None, beta[i][0], include, out, constr, names, 'initial')	

def add_other_constraints(constr,constraints,ll):
	for i in constraints:
		add_other_constraint(constr,i,ll)
		
		
def add_other_constraint(constr,constraint,ll):
	"""Adds a user constraint\n\n
		constraint shall be on the format (name, minimum_or_value, maximum,index)
		where maximum and index are not required. If maximum is not supplied, minimum_or_value
		is a binding constraint. index is  the sequence of the variable within the group given
		by name. A constraint on the 'beta' variables at position 2 is indicated by submitting 2
		as the index argument."""		
	constraint=(list(constraint)+[None]*2)[:4]#in case one or both last arguments are not submitted
	name, minimum_or_value, maximum,index=constraint
	if ((type(minimum_or_value)==str) or (type(maximum)==str)) and ll is None:
		return	
	elif type(minimum_or_value)==str:
		minimum_or_value=eval(minimum_or_value,globals(),ll.__dict__)
	elif type(maximum)==str:
		maximum=eval(maximum,globals(),ll.__dict__)		

	constr.set_named(name, minimum_or_value,maximum,index)



class output:
	def __init__(self):
		self.variable=[]
		self.set_to=[]
		self.assco=[]
		self.cause=[]

	def add(self,variable,set_to,assco,cause):
		if (not (variable in self.variable)) or (not (cause in self.cause)):
			self.variable.append(variable)
			if type(set_to)==str:
				self.set_to.append(set_to)
			else:
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


class constraints:

	"""Stores the constraints of the LL maximization"""
	def __init__(self,args,old_constr):
		self.constraints=dict()
		self.categories=[]
		self.associates=dict()
		self.args=args
		if old_constr is None:
			self.old_constr=[]
		else:
			self.old_constr=old_constr.constraints


	def set(self,positions, minimum_or_value,maximum=None,replace=True,assoc=None):
		"""Adds a constraint. 'positions' is either an integer or an iterable of integer specifying the position(s) 
		for which the constraints shall apply. If 'positions' is a string, it is assumed to be the name of a category \n\n

		Equality constraints are chosen by specifying 'minimum_or_value' \n\n
		Inequality constraints are chosen specifiying 'maximum' and 'minimum'\n\n
		'replace' determines whether an existing constraint shall be replaced or not 
		(only one equality and inequality allowed per position)"""
		if type(positions)==int or type(positions)==np.int64  or type(positions)==np.int32:
			positions=[positions]
		elif type(positions)==str:
			positions=self.args.positions[positions]
		for i in positions:
			if replace or (i not in self.constraints):
				if maximum==None:
					self.constraints[i]=[minimum_or_value]
				else:
					if minimum_or_value<maximum:
						self.constraints[i]=[minimum_or_value,maximum]
					else:
						self.constraints[i]=[maximum,minimum_or_value]
				if not assoc is None:
					self.associates[i]=assoc
			category=self.args.map_to_categories[i]
			if not category in self.categories:
				self.categories.append(category)

	def set_named(self,name,minimum_or_value,maximum=None,index=None):
		if index is None:
			positions=self.args.positions[name]
		else:
			positions=self.args.positions[name][index]
		self.set(positions, minimum_or_value,maximum)
		return
		
	def constraints_to_arrays(self):
		c=[]
		c_eq=[]
		for i in self.constraints:
			if len(self.constraints[i])==1:
				c_eq.append(self.constraints[i]+[i])
			else:
				c.append(self.constraints[i]+[i])
		return c,c_eq



