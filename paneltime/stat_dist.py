#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Generates numerical distribution functions that don't require the slow loading of scipy

from mpmath import *
import numpy as np

def tcdf(x, n):
	if type(x)==float or type(x)==np.float64:
		w = n/(x**2+n)
		I = betainc(0.5*n, 0.5,x2=w, regularized = True)
		dist = float(1-I)
		return dist
	if len(x)>100:
		raise RuntimeWarning(f"Dimension is {len(x)}. This function is not made for such large dimension. Consider using scipy. ")
	return np.array([tcdf(i, n) for i in x])
	
	
def norm(x,mu=0,s=1, cdf = True):
	if type(x)==float:
		if not cdf:
			return npdf(x, mu, s)
		f = 0.5*erfc((mu-x)/(s*2**0.5))
		return f
	if len(x)>100:
		raise RuntimeWarning(f"Dimension is {len(x)}. This function is not made for such large dimension. Consider using scipy. ")
	return np.array([norm(i, mu, s, cdf) for i in x])	
	
def chisq(x, k):
	if x<0:
		return 0
	c = gammainc(0.5*k, 0, 0.5*x)/gamma(0.5*k)
	return c

def fcdf(x, k1, k2):
	f = betainc(0.5*k1, 0.5*k2, x1 = 0, x2 = k1*x/(k1*x+k2), regularized = True)
	return f

def test_functions():
	from scipy import stats as scstats
	print(tcdf(3,5))
	print(scstats.t.cdf(3,5))
	
	print(norm(3,5,2))
	print(scstats.norm.cdf(3,5,2))
	
	print(chisq(3,5))
	print(scstats.chi2.cdf(3,5))
	
	print(fcdf(3,5,1))
	print(scstats.f.cdf(3,5,1))	
	
	print(norm(3,5,1, False))
	print(scstats.norm.pdf(3,5,1))		
	
#test_functions()
a=0