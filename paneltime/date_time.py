#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import functions as fu

monthdays=np.array([31,28,31,30,31,30,31,31,30,31,30,31]).reshape((1,12))
month_rng=np.arange(1,13).reshape((1,12))
monthdays=np.cumsum(monthdays,1)

def days_since(years=0,months=0,days=0,hours=0, minutes=0,seconds=0,base_year=2000):
	if (np.max(months)>12 or np.max(days)>31 
	    or np.min(days)<0 or  np.min(months)<0):
		raise RuntimeError("There is something wrong with a string variable that is formatted like a date-time variable")

	yr_rng=np.arange(min((np.min(years),base_year))-1,max((np.max(years),base_year)))
	n=len(days)
	k=len(yr_rng)
	yr_rng=yr_rng.reshape((1,k))
	
	#Calculating leap years
	leap_year=(yr_rng/4.0).astype(np.int)==yr_rng/4.0
	leap_year=leap_year*((yr_rng/100.0).astype(np.int)!=yr_rng/100.0)
	leap_year=(leap_year+((yr_rng/400.0).astype(np.int)==yr_rng/400.0))>0
	
	#Calculating number of days
	yr_sumdays=np.cumsum(leap_year+365,1)
	yr_days=np.sum((years-1==yr_rng)*yr_sumdays,1).reshape((n,1))
	month_days=np.sum((months-1==month_rng)*monthdays,1).reshape((n,1))
	month_days=month_days+(months>2)*np.sum((years==yr_rng)*leap_year,1).reshape((n,1))
	
	#subtracting january 1. of base year
	i=np.nonzero(yr_rng==base_year-1)[1][0]
	days_base_year=yr_sumdays[0,i]+1
	
	days_count=yr_days+month_days+days-days_base_year 
	days_count=days_count+time_fraction(hours, minutes, seconds)
	return days_count
	
	
def time_fraction(hours,minutes,seconds):
	if (np.max(hours)>24.0 or np.max(minutes)>60.0 or 
	    np.max(seconds)>60.0 or  np.min(hours)<0 
	    or np.min(minutes)<0 or  np.min(seconds)<0):
		raise RuntimeError("There is something wrong with a string variable that is formatted like a date-time variable")		
	return (hours/24.0)+(minutes/1440.0)+(seconds/86400.0)

	
def test():
	dates=np.loadtxt('../Input/date.csv',delimiter='.')
	days,months,years=dates[:,0:1],dates[:,1:2],dates[:,2:3]
	d=days_since_2000(years, months, days)
	fu.savevar(d,extension='csv')
	pass
	
	
#test()