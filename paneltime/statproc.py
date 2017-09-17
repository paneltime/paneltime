#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains statistical procedures

from scipy import stats as scstats
from scipy import special as sc
import regprocs as rp
import numpy as np
import regprocs as rp
import functions as fu
import loglikelihood as logl


def var_decomposition(XXNorm=None,X=None,concat=False):
	"""Variance decomposition. Returns the matrix of condition indexes for each factor (rows) and each variable
	(columns). Calculates the normalized sum of squares using square_and_norm if XXNorm is not supplied"""
	if XXNorm is None:
		XXNorm=square_and_norm(X)
	ub=len(XXNorm)     
	d,EVec=np.linalg.eig(XXNorm)
	if np.any(np.round(d.imag,15)!=len(d)*[0]):
		print( "non-real XX matrix")
		
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
	norm=fu.dot(X,X)/(Sumsq+1e-200)
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
	Singularity is of course an extreme form of multicollinearity. This functino is needed 
	since the nummerical procedure for detecting mc does not handle singularity"""
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
	beta,y=OLS(panel,panel.X*panel.included,panel.Y*panel.included,return_e=True)
	y=ll.re_obj.FRE(y)
	y_dev=deviation(panel,y)
	s=panel.var(y_dev,1)**0.5
	s=s.reshape(N,1,1)
	y=y/(s+(s==0)*1e-17)
	yl1=rp.roll(y,1,1)
	dy=y-yl1
	date_var=np.arange(T).reshape((T,1))*panel.included	#date count
	X=np.concatenate((panel.included,date_var,yl1),2)
	dyL=[]
	for i in range(p):
		dyL.append(rp.roll(dy,i+1,1))
	dyL=np.concatenate(dyL,2)
	date_var=(date_var>p+1)
	X=np.concatenate((X,dyL),2)
	X=X*date_var
	dy=dy*date_var
	X[:,0:panel.lost_obs+10]=0
	keep,c_ix=singular_elim(panel,X)
	if not np.all(keep[0:3]):
		return 'NA'
	beta,se=OLS(panel,X[:,:,keep],dy,return_se=True,c=date_var)
	adf_stat=beta[2]/se[2]
	critval=adf_crit_values(panel.NT,True)
	res=np.append(adf_stat,critval)
	return res

def goodness_of_fit(panel,ll):
	v0=panel.var(ll.e_st)
	y=deviation(panel,ll.Y_st)
	v1=panel.var(y)
	Rsq=1-v0/v1
	Rsqadj=1-(v0/v1)*(panel.NT-1)/(panel.NT-panel.len_args-1)
	LL_OLS=logl.LL(panel.args.args_OLS,panel)
	if not LL_OLS is None:
		LL_OLS=LL_OLS.LL
		LL_ratio_OLS=2*(ll.LL-LL_OLS)
	else:
		LL_ratio_OLS=None
	LL_args_restricted=logl.LL(panel.args.args_restricted,panel)
	if not LL_args_restricted is None:
		LL_args_restricted=LL_args_restricted.LL
		LL_ratio=2*(ll.LL-LL_args_restricted)
	else:
		LL_ratio=None
	return Rsq, Rsqadj, LL_ratio,LL_ratio_OLS


def breusch_godfrey_test(panel,ll, lags):
	"""returns the probability that err_vec are not auto correlated""" 
	e=ll.e_st
	X=ll.X_st
	N,T,K=X.shape	
	X_u=X[:,lags:T]
	u=e[:,lags:T]
	c=panel.included[:,lags:T]
	for i in range(1,lags+1):
		X_u=np.append(X_u,e[:,lags-i:T-i],2)
	XX=fu.dot(X_u,X_u)
	Beta,Rsq=OLS(panel,X_u,u,True,True,c=c)
	T=(panel.NT-X_u.shape[2]-lags)
	BGStat=T*Rsq
	rho=Beta[len(X[0]):]
	ProbNoAC=1.0-chisq_dist(BGStat,lags)
	print( 'BG-test R-squared: %s  Prob. no AC: %s' %(Rsq,ProbNoAC))
	return ProbNoAC, rho, Rsq #The probability of no AC given H0 of AC.


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
	N,T,k=e.shape
	e=e*panel.included
	df=panel.NT
	s=(np.sum(e**2)/df)**0.5
	mu3=np.sum(e**3)/df
	mu4=np.sum(e**4)/df
	S=mu3/s**3
	C=mu4/s**4
	JB=df*((S**2)+0.25*(C-3)**2)/6.0
	return 1.0-chisq_dist(JB,2)


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
	cov=fu.dot(X,X)/N
	
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

def OLS(panel,X,Y,add_const=False,return_rsq=False,return_e=False,c=None,return_se=False):
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
	XX=fu.dot(X,X)
	XXInv=np.linalg.inv(XX)
	XY=fu.dot(X,Y)
	beta=fu.dot(XXInv,XY)

	if return_rsq or return_e or return_se:
		e=Y-fu.dot(X,beta)
		if return_rsq:
			v0=np.var(e)
			v1=np.var(Y)
			Rsq=1-v0/v1
			#Rsqadj=1-(v0/v1)*(NT-1)/(NT-k-1)
			return beta,Rsq
		elif return_e:
			return beta,e
		elif return_se:
			se=np.std(e)*(np.diag(XXInv)**0.5)
			return beta,se.reshape(k,1)
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

def newey_west_wghts(L,X=None,err_vec=None,XErr=None):
	"""Calculates the Newey-West autocorrelation consistent weighting matrix. Either err_vec or XErr is required"""
	if XErr is None:
		XErr=X*err_vec
	N,T,k=XErr.shape
	S=fu.dot(XErr,XErr)#Whites heteroscedasticity consistent weighting matrix
	for i in range(1,min(L,T)):
		w=1-(i+1)/(L)
		S+=w*fu.dot(XErr[:,i:],XErr[:,0:T-i])
		S+=w*fu.dot(XErr[:,0:T-i],XErr[:,i:])
	return S






