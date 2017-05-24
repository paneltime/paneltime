#!/usr/bin/env python
# -*- coding: utf-8 -*-

#contains the log likelihood object

import numpy as np
import functions as fu
import regprocs as rp
import statproc as stat
import calculus

class LL:
	"""Calculates the log likelihood given arguments arg (either in dictonary or array form), and store all 
	associated dynamic variables needed outside this scope"""
	def __init__(self,args,panel,X=None):

		if args is None:
			args=panel.args.args
		self.LL_const=-0.5*np.log(2*np.pi)*panel.NT_afterloss
	
		self.args_v=panel.args.conv_to_vector(panel,args)
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
		AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA=matrices
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
			avg_h=(np.sum(h_val,1)/panel.T_arr).reshape((N,1,1))*panel.a
			if panel.N>1:
				lnv=lnv+args['mu'][0]*avg_h
			lnv=np.maximum(np.minimum(lnv,100),-100)
		v=np.exp(lnv)*panel.a
		v_inv=np.exp(-lnv)*panel.a	
		e_RE=rp.RE(self,panel,e)
		e_REsq=e_RE**2
		LL=self.LL_const-0.5*np.sum((lnv+(e_REsq)*v_inv)*panel.included)
		
		if abs(LL)>1e+100: 
			return None
		self.AMA_1,self.AAR,self.AMA_1AR,self.GAR_1,self.GMA,self.GAR_1MA=matrices
		self.u,self.e,self.h_e_val,self.h_val, self.lnv_ARMA        = u,e,h_e_val,h_val, lnv_ARMA
		self.lnv,self.avg_h,self.v,self.v_inv,self.e_RE,self.e_REsq = lnv,avg_h,v,v_inv,e_RE,e_REsq
		self.h_2e_val,self.h_z_val,self.h_ez_val,self.h_2z_val      = h_2e_val,h_z_val,h_ez_val,h_2z_val
		self.e_st=e_RE*v_inv
		
		return LL
	

	def standardize(self,panel):
		"""Adds X and Y and error terms after ARIMA-E-GARCH transformation and random effects to self"""
		v_inv=self.v_inv**0.5
		m=panel.lost_obs
		N,T,k=panel.X.shape
		Y=fu.dot(self.AMA_1AR,panel.Y)
		Y=rp.RE(self,panel,Y,False)*v_inv
		X=fu.dot(self.AMA_1AR,panel.X)
		X=rp.RE(self,panel,X,False)*v_inv
		self.e_st=self.e_RE*v_inv
		self.Y_st=Y
		self.X_st=X
		self.e_st_long=panel.de_arrayize(self.e_st,m)
		self.Y_st_long=panel.de_arrayize(self.Y_st,m)
		self.X_st_long=panel.de_arrayize(self.X_st,m)

	def copy_args_d(self):
		return fu.copy_array_dict(self.args_d)

	
	def h(self,e,z):
		d={'e':e,'z':z}
		try:
			exec(self.h_def,globals(),d)
		except Exception as err:
			if self.h_err!=str(err):
				print ("Warning: error in the ARCH error function h(e,z). The error was: %s" %(err))
			h_err=str(e)
			return None
	
		return d['ret']	
	


def set_garch_arch(panel,args):
	if panel.max_T>50000:#For really large matrices, sparse inversion is faster and less memory consuming
		return set_garch_arch_sparse(panel,args)

	p,q,m,k,nW=panel.p,panel.q,panel.m,panel.k,panel.nW
	X=panel.I+lag_matr(panel.L,panel.zero,q,args['lambda'])
	try:
		AMA_1=np.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(AMA_1)):
		return None
	AAR=panel.I-lag_matr(panel.L,panel.zero,p,args['rho'])
	AMA_1AR=fu.dot(AMA_1,AAR)
	X=panel.I-lag_matr(panel.L,panel.zero,k,args['gamma'])
	try:
		GAR_1=np.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(GAR_1)):
		return None		
	GMA=lag_matr(panel.L,panel.zero,m,args['psi'])	
	GAR_1MA=fu.dot(GAR_1,GMA)
	return AMA_1,AAR,AMA_1AR,GAR_1,GMA,GAR_1MA



def set_garch_arch_sparse(panel,args):
	p,q,m,k,nW=panel.p,panel.q,panel.m,panel.k,panel.nW

	X=panel.I_sp+lag_matr_sp(q,args['lambda'])
	try:
		AMA_1=sp.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(AMA_1.data)):
		return None
	AAR=panel.I_sp-lag_matr_sp(p,args['rho'])
	AMA_1AR=fu.dot(AMA_1,AAR)
	X=panel.I_sp-lag_matr_sp(k,args['gamma'])
	try:
		GAR_1=sp.linalg.inv(X)
	except:
		return None
	if np.any(np.isnan(GAR_1.data)):
		return None		
	GMA=lag_matr_sp(m,args['psi'])
	GAR_1MA=fu.dot(GAR_1,GMA)

	return AMA_1.toarray(),AAR.toarray(),AMA_1AR.toarray(),GAR_1.toarray(),GMA.toarray(),GAR_1MA.toarray()



def lag_matr_sp(panel,k,args):
	T=panel.max_T
	M=sp.csc_matrix(([],([],[])),(T,T))
	for i in range(k):
		M=M+args[i]*panel.L_sp[i]
	return M


def lag_matr(L,zero,k,args):
	if k==0:
		return zero
	a=[]
	for i in range(k):
		a.append(args[i]*L[i])
	a=np.sum(a,0)
	return a


class direction:
	def __init__(self,panel):
		self.gradient=calculus.gradient(panel)
		self.hessian=calculus.hessian(panel)
		self.panel=panel
		self.constr=4*[None]
		self.hessin_num=np.diag(np.ones(panel.args.n_args))
		self.g_old=None
		
		
	def get(self,ll,mc_limit,dx_conv,k,its,mp=None,dxi=None,print_on=True):

		g,G=self.gradient.get(ll,return_G=True)		
		hessians=self.get_hessians(ll,mp,g,G,dxi)
		
		dx_list=[]
		a=[]

		for i in range(len(hessians)):
			if hessians[i][0] is None:
				dx_list.append([None,hessians[i][1],None])
				a.append([None,None])
			else:
				out=output(print_on)
				self.constr[i]=constraints(self.panel.args,self.constr[i])
				hessian,reset=add_constraints(G,self.panel,ll,self.constr[i],mc_limit,dx_conv,hessians[i][0],k,its,out)
				dc,constrained=solve(self.constr[i],hessian, g, ll.args_v)
				dc_tmp=dc*1
				for j in range(len(dc)):
				
					s=dc*(constrained==0)*g
					if np.sum(s)<0:#negative slope
						s=np.argsort(s)
						remove(s[0], None, ll.args_v, None, out, self.constr[i], self.panel.name_vector, 'neg. slope')
						dc,constrained=solve(self.constr[i],hessian, g, ll.args_v)
					else:
						break
				#if np.all(dc==0):
				#	dc=np.sign(g)*(np.abs(dc)==np.max(np.abs(dc)))
				dx_list.append([dc,hessians[i][1],out])
				a.append([constrained,reset])
		
		#the analytical solution at position 0 is the one returned and tested for convergence
		dc,name,out=dx_list[0]
		constrained,reset=a[0]
		s=dc*(constrained==0)*g
		if  np.sum(s)<0:
			dc,name,out=dx_list[1]
			constrained,reset=a[1]			
		return dc,g,G,hessians[0][0],constrained,reset
	

	
	def get_hessians(self,ll,mp,g,G,dxi):
		hessian=self.hessian.get(ll,mp)
		hessinS0=rp.sandwich(hessian,G,0)
		hessinS100=rp.sandwich(hessian,G,100)

		
		if (not self.g_old is None) and (not dxi is None):
			self.hessin_num=approximate_hessin(g,self.g_old,self.hessin_num,dxi)	
		for i in range(len(hessinS0)):
			hessinS0[i,i]=hessinS0[i,i]+(hessinS0[i,i]==0)	
			hessinS100[i,i]=hessinS100[i,i]+(hessinS100[i,i]==0)
			self.hessin_num[i,i]=self.hessin_num[i,i]+(self.hessin_num[i,i]==0)
		try:
			hessinS0=-np.linalg.inv(hessinS0)
			hessS100=-np.linalg.inv(hessinS100)
			hess_num=-np.linalg.inv(self.hessin_num)
		except:
			pass
		self.g_old=g
		
		hessians=[
		[hessian,		'analytic'],
		[hessinS0,		'analytic sandwich 0 lags'],
		[hessinS100,	'analytic sandwich 100 lags'],
		[hess_num,	'numerical']
		]

		return hessians
	
def approximate_hessin(g,g_old,hessin,dxi):
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
	v=stat.var(panel,G)
	for i in range(1,k):
		if v[0][i]==0:
			include[i]=False
			constr.add(i,0)
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


def remove_one_multicoll(G,args,names,include,out,constr,limit):
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
	constr.add(d,a)
	if not include is None:
		include[d]=False	
	if not assoc is None:
		out.add(names[d],a,names[assoc],r_type)	
	else:
		out.add(names[d],a,'NA',r_type)	
	return True

def add_constraints(G,panel,ll,constr,mc_limit,dx_conv,hessian,k,its,out):
	names=panel.name_vector
	args=ll.args_v
	N,T,h=G.shape
	include=np.ones(h,dtype=bool)
	add_initial_constraints(panel,constr,out,names,ll,include,its)
	remove_constants(panel, G, include,constr,out,names)	
	remove_all_multicoll(G, args, names, include, out, constr, 1000)
	reset=False
	#remove_H_correl(hessian,include,constr,args,out,names)
	if mc_limit<30 and not (dx_conv is None):
		srt=np.argsort(dx_conv)
		for i in range(min((k,len(srt)-2))):
			j=srt[-i-1]
			if dx_conv[j]<0.05:
				reset=True
			else:
				reset=remove(j,None,args, include, out,constr,names,'dir cap')==False
	return hessian, reset

def add_initial_constraints(panel,constr,out,names,ll,include,its):
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




class output:
	def __init__(self,on=True):
		self.variable=[]
		self.set_to=[]
		self.assco=[]
		self.cause=[]
		self.on=on

	def add(self,variable,set_to,assco,cause):
		if (not (variable in self.variable)) or (not (cause in self.cause)):
			self.variable.append(variable)
			self.set_to.append(str(round(set_to,8)))
			self.assco.append(assco)
			self.cause.append(cause)

	def print(self):
		if self.on==False:
			return
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
		if self.on:
			print(output)	


class constraints:

	"""Stores the constraints of the LL maximization"""
	def __init__(self,args,old_constr):
		self.constraints=dict()
		self.categories=[]
		self.args=args
		if old_constr is None:
			self.old_constr=[]
		else:
			self.old_constr=old_constr.constraints


	def add(self,positions, minimum_or_value,maximum=None,replace=True):
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
			category=self.args.map_to_categories[i]
			if not category in self.categories:
				self.categories.append(category)


	def constraints_to_arrays(self):
		c=[]
		c_eq=[]
		for i in self.constraints:
			if len(self.constraints[i])==1:
				c_eq.append(self.constraints[i]+[i])
			else:
				c.append(self.constraints[i]+[i])
		return c,c_eq

	def remove(self):
		"""Removes arbitrary constraint"""
		k=list(self.constraints.keys())[0]
		self.constraints.pop(k)



