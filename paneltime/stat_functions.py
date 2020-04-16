#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains statistical procedures

from scipy import stats as scstats
from scipy import special as sc
import calculus_functions as cf
import numpy as np
import loglikelihood as logl
import random_effects


def var_decomposition(XXNorm=None,X=None,concat=False):
	"""Variance decomposition. Returns the matrix of condition indexes for each factor (rows) and each variable
	(columns). Calculates the normalized sum of squares using square_and_norm if XXNorm is not supplied"""
	if XXNorm is None:
		XXNorm=square_and_norm(X)
	ub=len(XXNorm)     
	d,EVec=np.linalg.eig(XXNorm)
	if np.any(np.round(d.imag,15)!=len(d)*[0]):
		pass
		#print( "non-real XX matrix")
		
	d=d.real;EVec=EVec.real
	d=np.abs(d)**0.5+1e-100
	MaxEv=np.max(d)  
	fi=np.abs(EVec*EVec/((d*d).reshape((1,ub))+1E-200))
	fiTot=np.sum(fi,1)
	pi=fi.transpose()/fiTot.reshape((1,ub))
	CondIx=MaxEv/d
	ind=np.argsort(CondIx)
	pi=pi[ind]
	CondIx=CondIx[ind]
	CondIx=CondIx.reshape((len(CondIx),1))
	if concat:
		return np.concatenate((CondIx,pi),1)
	else:
		return CondIx,pi

def square_and_norm(X):
	"""Squares X, and normalize to unit lenght.
	Similar to a correlation matrix, except the
	means are not subtracted"""
	N,T,k=X.shape
	Sumsq=np.sqrt(np.sum(np.sum(X**2,0),0))
	Sumsq.resize((k,1))
	Sumsq=Sumsq*Sumsq.T
	norm=cf.dot(X,X)/(Sumsq+1e-200)
	return norm

def singular_elim(panel,X):
	"""Eliminates variables that cause singularity"""
	N,T,k=X.shape
	r=np.arange(k)
	ci_threshold=50
	keep,XXCorrel=find_singulars(panel,X) 
	XXNorm=square_and_norm(X)
	cond_ix,pi=var_decomposition(XXNorm)
	for i in range(0,k):
		if np.sum(keep)<2:
			break
		XXtmp=XXNorm[keep][:,keep]
		keep_map=r[keep]
		cond_ix_tmp,pi_tmp=var_decomposition(XXtmp)
		if cond_ix_tmp is None:
			break
		if cond_ix_tmp[len(XXtmp)-1]<ci_threshold:
			break			
		cond_ix=np.zeros((k,1))
		pi[keep[:,None]*keep[None,:]]=pi_tmp.flatten()
		if not np.all(pi[keep][:,keep]==pi_tmp):
			raise RuntimeError('OK, so it did not work allways')
		cond_ix[keep_map]=cond_ix_tmp

		nz=np.nonzero((pi>0.5)*(cond_ix>ci_threshold)+(pi==np.max(pi,1))*(cond_ix>1000))#More than a 50% variance proportion is "high", single items with more than 1000 are also deleted
		drop_items=np.append(nz[0][1:]==nz[0][:len(nz[1])-1],False)#deleting only k-1 of all k collienar observations
		keep[nz[1][drop_items]]=False
		keep[0]=True#Never remove the constant term
	return keep,cond_ix

def get_full_reg(beta,se,keep):
	"fills in zeroes where variables have been deleted by singular_elim"
	k=len(keep)
	beta_full=np.zeros((k,1))
	ret[keep]=beta	
	se_full=np.zeros((k,1))
	ret[keep]=se	
	return ret


def find_singulars(panel,X):
	"""Returns a list with True for variables that cause singularity and False otherwise.
	for the main regression, singularity is detected by the constraints module"""
	N,T,k=X.shape
	XXCorrel=correl(X,panel)
	keep=np.all(np.isnan(XXCorrel)==False,0)
	keep=keep*np.all((np.abs(np.triu(XXCorrel,1))>0.99)==False,0)
	x_dev=deviation(panel, X)
	var_x=np.sum(np.sum(x_dev**2,0),0)
	keep=keep*(var_x>0)#remove constants
	keep[0]=True#allways keep the first constant term
	return keep,XXCorrel

def adf_test(panel,ll,p):
	"""Returns the augmented dickey fuller test statistic and critical value"""
	N,T,k=panel.X.shape
	y=ll.Y_st
	yl1=cf.roll(y,1,1)
	dy=y-yl1
	date_var=np.arange(T).reshape((T,1))*panel.included	#date count
	X=np.concatenate((panel.included,date_var,yl1),2)
	dyL=[]
	for i in range(p):
		dyL.append(cf.roll(dy,i+1,1))
	dyL=np.concatenate(dyL,2)
	date_var=(date_var>p+1)
	X=np.concatenate((X,dyL),2)
	X=X*date_var
	dy=dy*date_var
	X[:,0:panel.lost_obs+10]=0
	keep,c_ix=singular_elim(panel,X)
	if not np.all(keep[0:3]):
		return 'NA','NA','NA'
	beta,se_robust,se=OLS(panel,X[:,:,keep],dy,robust_se_lags=10,c=date_var)
	adf_stat=beta[2]/se_robust[2]
	critval=adf_crit_values(panel.NT,True)
	res=np.append(adf_stat,critval)
	return res

def goodness_of_fit(panel,ll):
	s_res=panel.var(ll.e_st)
	s_mod=panel.var(ll.Y_pred_st)
	s_expl=s_res/(s_mod+s_res)
	Rsq=1-s_expl
	Rsqadj=1-s_expl*(panel.NT-1)/(panel.NT-panel.args.n_args-1)
	panel.args.create_null_ll()
	LL_ratio_OLS=2*(ll.LL-panel.args.LL_OLS)
	LL_ratio=2*(ll.LL-panel.args.LL_null)
	return Rsq, Rsqadj, LL_ratio,LL_ratio_OLS


def breusch_godfrey_test(panel,ll, lags):
	"""returns the probability that err_vec are not auto correlated""" 
	e=ll.e_st_centered
	X=ll.X_st
	N,T,k=X.shape
	X_u=X[:,lags:T]
	u=e[:,lags:T]
	c=panel.included[:,lags:T]
	for i in range(1,lags+1):
		X_u=np.append(X_u,e[:,lags-i:T-i],2)
	Beta,Rsq=OLS(panel,X_u,u,False,True,c=c)
	T=(panel.NT-k-1-lags)
	BGStat=T*Rsq
	rho=Beta[k:]
	ProbNoAC=1.0-chisq_dist(BGStat,lags)
	return ProbNoAC, rho, Rsq #The probability of no AC given H0 of AC.

def correlogram(panel,e,lags):
	"""returns the probability that err_vec are not auto correlated""" 
	N,T,k=e.shape	
	v=panel.var(e)
	rho=np.zeros(lags+1)
	rho[0]=1
	for i in range(1,lags+1):
		a=panel.T_i-i-1>0
		df=np.sum(a*(panel.T_i-i-1))
		rho[i]=np.sum(a*e[:,i:]*e[:,0:-i])/(v*df)
	return rho #The probability of no AC given H0 of AC.

def chisq_dist(X,df):
	"""Returns the probability of drawing a number
	less than X from a chi-square distribution with 
	df degrees of freedom"""
	retval=1.0-sc.gammaincc(df/2.0,X/2.0)
	return retval


def adf_crit_values(n,trend):
	"""Returns 1 and 5 percent critical values respectively"""
	if trend:
		d={25:np.array([-3.75,-3.00]),50:np.array([-3.58,-2.93]),100:np.array([-3.51,-2.89]),
		   250:np.array([-3.46,-2.88]),500:np.array([-3.44,-2.87]),10000:np.array([-3.43,-2.86])}
	else:
		d={25:np.array([-4.38,-3.60]),50:np.array([-4.15,-3.50]),100:np.array([-4.04,-3.45]),
		   250:np.array([-3.99,-3.43]),500:np.array([-3.98,-3.42]),10000:np.array([-3.96,-3.41])}

	if n<25:
		print ("Warning: ADF critical values are not available for fewer than 25 observations")
		return (0,0)
	k=(25,50,100,250,500,10000)
	r=None
	for i in range(len(k)-1):
		if n>=k[i] and n<500:
			r=d[k[i]]+(n-k[i])*((d[k[i+1]]-d[k[i]])/(k[i+1]-k[i]))#interpolation
			return r
	if r is None:
		return d[10000]

def JB_normality_test(e,panel):
	"""Jarque-Bera test for normality. 
	returns the probability that a set of residuals are drawn from a normal distribution"""
	e=e[panel.included]
	a=np.argsort(np.abs(e))[::-1]
	
	ec=e[a][int(0.001*len(e)):]
	
	df=len(ec)
	ec=ec-np.mean(ec)
	s=(np.sum(ec**2)/df)**0.5
	mu3=np.sum(ec**3)/df
	mu4=np.sum(ec**4)/df
	S=mu3/s**3
	C=mu4/s**4
	JB=df*((S**2)+0.25*(C-3)**2)/6.0
	p=1.0-chisq_dist(JB,2)
	return p


def correl(X,panel=None):
	"""Returns the correlation of X and Y. Assumes three dimensional matrixes. If Y is not supplied, the 
	correlation matrix of X is returned"""
	if not panel is None:
		X=X*panel.included
		N,T,k=X.shape
		N=panel.NT
		mean=np.sum(np.sum(X,0),0).reshape((1,k))/N
	else:
		N,k=X.shape
		mean=np.sum(X,0).reshape((1,k))/N
	cov=cf.dot(X,X)/N
	
	cov=cov-(mean.T*mean)
	stdx=(np.diag(cov)**0.5).reshape((1,k))
	stdx=(stdx.T*stdx)
	corr=(stdx>0)*cov/(stdx+(stdx==0)*1e-100)
	
	return corr

def deviation(panel,X):
	N,T,k=X.shape
	x=X*panel.included
	mean=np.sum(np.sum(x,0),0).reshape((1,1,k))/panel.NT
	return (X-mean)*panel.included

def correl_2dim(X,Y=None):
	"""Returns the correlation of X and Y. Assumes two dimensional matrixes. If Y is not supplied, the 
	correlation matrix of X is returned"""	
	if type(X)==list:
		X=Concat(X)
	single=Y is None
	if single:
		Y=X
	T,k=X.shape
	T,m=Y.shape
	X_dev=X-np.mean(X,0)
	Y_dev=Y-np.mean(Y,0)
	cov=np.dot(X_dev.T,Y_dev)
	stdx=np.sum(X_dev**2,0).reshape((1,k))**0.5
	if single:
		stdy=stdx
	else:
		stdy=np.sum(Y_dev**2,0).reshape((1,k))**0.5
	std_matr=stdx.T*stdy
	std_matr=std_matr+(std_matr==0)*1e-200
	corr=cov/std_matr
	if corr.shape==(1,1): 
		corr=corr[0][0]
	return cov/std_matr

def OLS(panel,X,Y,add_const=False,return_rsq=False,return_e=False,c=None,robust_se_lags=0):
	"""runs OLS after adding const as the last variable"""
	if c is None:
		c=panel.included
	N,T,k=X.shape
	NT=panel.NT
	if add_const:
		X=np.concatenate((c,X),2)
		k=k+1
	X=X*c
	Y=Y*c
	XX=cf.dot(X,X)
	XY=cf.dot(X,Y)
	beta=np.linalg.solve(XX,XY)
	if return_rsq or return_e or robust_se_lags:
		e=(Y-cf.dot(X,beta))*c
		if return_rsq:
			v0=panel.var(e,included=c)
			v1=panel.var(Y,included=c)
			Rsq=1-v0/v1
			#Rsqadj=1-(v0/v1)*(NT-1)/(NT-k-1)
			return beta,Rsq
		elif return_e:
			return beta,e*c
		elif robust_se_lags:
			XXInv=np.linalg.inv(XX)
			se_robust,se=robust_se(panel,robust_se_lags,XXInv,X*e)
			return beta,se_robust.reshape(k,1),se.reshape(k,1)
	return beta

def OLS_simple(Y,X,addconst=False,residuals=True):
	"""Returns the OLS residuals if residuals. For use with two dimiensional arrays only"""
	if addconst:
		n=len(X)
		X=np.concatenate((np.ones((n,1)),X),1)
	XY=np.dot(X.T,Y)
	XX=np.dot(X.T,X)
	XXInv=np.linalg.inv(XX)
	beta=np.dot(XXInv,XY)
	e=Y-np.dot(X,beta)
	if residuals:
		return beta,e
	else:
		return beta

def newey_west_wghts(L,XErr):
	"""Calculates the Newey-West autocorrelation consistent weighting matrix. Either err_vec or XErr is required"""
	N,T,k=XErr.shape
	S=np.zeros((k,k))
	try:
		a=min(L,T)
	except:
		a=0
	for i in range(1,min(L,T)):
		w=1-(i+1)/(L)
		XX=cf.dot(XErr[:,i:],XErr[:,0:T-i])
		S+=w*(XX+XX.T)
	return S

def robust_cluster_weights(panel,XErr,cluster_dim,whites):
	"""Calculates the Newey-West autocorrelation consistent weighting matrix. Either err_vec or XErr is required"""
	N,T,k=XErr.shape
	if cluster_dim==0:#group cluster
		if N<=1:
			return 0
		mean=panel.mean(XErr,0)
	elif cluster_dim==1:#time cluster
		mean=random_effects.mean_time(panel,XErr,True)
		T,m,k=mean.shape
		mean=mean.reshape((T,k))
	S=cf.dot(mean,mean)-whites
	return S


def robust_se(panel,L,hessin,XErr,nw_only=True):
	"""Returns the maximum robust standard errors considering all combinations of sums of different combinations
	of clusters and newy-west"""
	w=sandwich_var(hessin,cf.dot(XErr,XErr))#whites
	nw=sandwich_var(hessin,newey_west_wghts(L,XErr))#newy-west
	c0=sandwich_var(hessin,robust_cluster_weights(panel,XErr, 0, w))#cluster dim 1
	c1=sandwich_var(hessin,robust_cluster_weights(panel,XErr, 1, w))#cluster dim 2
	v=np.array([
		w*0,
		nw,
		nw+c0,
		nw+c1,
		nw+c1+c0
	])
	se_robust=np.maximum(np.max(w+v,0),0)**0.5
	se_std=np.maximum(w,0)**0.5
	return se_robust,se_std
	
def sandwich_var(hessin,V):
	hessinV=cf.dot(hessin,V)
	v=cf.dot(hessinV,hessin)
	v=np.diag(v)
	return v
	



