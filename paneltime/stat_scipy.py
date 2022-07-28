#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module contains statistical procedures

from scipy import special as sc
import stat_functions as stat
import numpy as np

def goodness_of_fit(ll,standarized,panel):
	if standarized:
		s_res=panel.var(ll.e_RE)
		s_tot=panel.var(ll.Y_st)
	else:
		s_res=panel.var(ll.u)
		s_tot=panel.var(panel.Y)		
	r_unexpl=s_res/s_tot
	Rsq=1-r_unexpl
	Rsqadj=1-r_unexpl*(panel.NT-1)/(panel.NT-panel.args.n_args-1)
	panel.args.create_null_ll(panel)
	LL_ratio_OLS=2*(ll.LL-panel.args.LL_OLS)
	LL_ratio=2*(ll.LL-panel.args.LL_null)
	return Rsq, Rsqadj, LL_ratio,LL_ratio_OLS

def breusch_godfrey_test(panel,ll, lags):
	"""returns the probability that err_vec are not auto correlated""" 
	e=ll.e_norm_centered
	X=ll.XIV_st
	N,T,k=X.shape
	X_u=X[:,lags:T]
	u=e[:,lags:T]
	c=panel.included[3][:,lags:T]
	for i in range(1,lags+1):
		X_u=np.append(X_u,e[:,lags-i:T-i],2)
	Beta,Rsq=stat.OLS(panel,X_u,u,False,True,c=c)
	T=(panel.NT-k-1-lags)
	BGStat=T*Rsq
	rho=Beta[k:]
	ProbNoAC=1.0-chisq_dist(BGStat,lags)
	return ProbNoAC, rho, Rsq #The probability of no AC given H0 of AC.

def chisq_dist(X,df):
	"""Returns the probability of drawing a number
	less than X from a chi-square distribution with 
	df degrees of freedom"""
	retval=1.0-sc.gammaincc(df/2.0,X/2.0)
	return retval



def JB_normality_test(e,panel):
	"""Jarque-Bera test for normality. 
	returns the probability that a set of residuals are drawn from a normal distribution"""
	e=e[panel.included[3]]
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


