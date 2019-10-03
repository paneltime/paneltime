#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import loglikelihood as logl
import os
import time
import output
import sys

digits_precision=12


def lnsrch(args, g, dx,panel,constr):
	
	f0=logl.LL(args, panel,constr)

	s=g*dx
	slope=np.sum(s)					#ensuring positive slope (should not be negative unless errors in gradient and/or hessian)
	if slope<=0 and False:
		return f0,"Linesearch roundoff problem"
	m=0.25
	for i in range(15+len(dx)):#Setting lmda so that the largest step is valid. Set ll.LL to return None when input is invalid
		lmda=m**i #Always try full Newton step first.
		if i>14:
			dx=dx*(np.abs(dx)<max(np.abs(dx)))
		x=f0.args_v+lmda*dx
		f1=logl.LL(x,panel,constraints=constr)
		if not f1.LL is None:
			break
	if i==14+len(x):
		return f0,'No valid values within newton step in linesearch'
	i=0
	d={f0.LL:f0,f1.LL:f1}

	while 1:
		i+=1
		f05=logl.LL(f0.args_v+lmda*(0.5**i)*dx,panel,constraints=constr)
		if not f05.LL is None:		
			break
	d[f05.LL]=f05
	for i in []:
		fcheck=logl.LL(f0.args_v+lmda*i*dx,panel,constraints=constr)
		if not fcheck.LL is None:
			d[fcheck.LL]=fcheck
	b=-(4*(f0.LL-f05.LL)+(f1.LL-f0.LL))/lmda
	c=2*((f0.LL-f05.LL)+(f1.LL-f05.LL))/(lmda**2)
	lambda_pred=lmda*0.25
	if c<0 and b>0:#concave increasing function
		lambda_pred=-b/(2*c)
		f_pred=logl.LL(f0.args_v+lambda_pred*dx,panel,constraints=constr) 
		if not f_pred.LL is None:	
			d[f_pred.LL]=f_pred
	
	f_max=max(d.keys())
	if f_max==f0.LL:#the function has not increased
		for j in range(1,6):
			s=(0.05**j)
			lmda=lambda_pred*s
			ll=logl.LL(f0.args_v+lmda*dx,panel,constraints=constr) 
			if ll.LL is None:
				break
			if ll.LL>f0.LL:
				return ll, "Newton step in linesearch to big, found an increment at %s of Newton step" %(s,)
	else:
		return d[f_max],'Linesearch success'
	return f0,'No increase in linesearch'#should never happen	
		
		
def maximize(panel,direction,mp,args_archive,args,_print,window):
	"""Maxmizes logl.LL"""

	its, convergence_limit   = 0, 0.001
	k, m, dx_norm            = 0,     0,    None
	H, prtstr, dxi           = None, '',None
	g       = None, False
	direction.hessin_num, ll = None, None
	n						 = panel.args.n_args
	while 1:  
		
		dx,g,G,H,constraints,ll=direction.get(ll,args,dx_norm,its,mp,dxi,False,k)
		f0=ll
		LL0=round_sign(ll.LL,digits_precision)
		dx_norm=direction.normalize(dx,ll.args_v)#np.abs(g/(np.diag(H)+(np.diag(H)==0)))#
		
			
		#Convergence test:
		lmt=convergence_limit*max(5*min((k,40)),1) 
		if np.max(np.abs(dx_norm)) < lmt and (its>4):  #max direction smaller than convergence_limit -> covergence
			#if m==3:
			if _print: print("Convergence on zero gradient; maximum identified")
			prtstr=printout(_print, ll, dx_norm,panel,its+1,constraints,"Convergence on zero gradient; maximum identified",window,H,G,direction.CI)
			return ll,g,G,H,1,prtstr,constraints,dx_norm
			#m+=1
			#precise_hessian=precise_hessian==False
		else:
			m=0
		
		prtstr=printout(_print, ll, dx_norm,panel,its+1,constraints,'linesearch on new direction',window,H,G,direction.CI)
		ll,msg=lnsrch(ll.args_d,g,dx,panel,constraints) 
		prtstr=printout(_print, ll, dx_norm,panel,its+1,constraints,msg,window,H,G,direction.CI)
		
		if window.finalized:
			print("Aborted")
			return ll,g,G,H, 0 ,prtstr,constraints,dx_norm

		args_archive.save(ll.args_d,0,(panel.p,panel.q,panel.m,panel.k,panel.d))
		
		dxi=f0.args_v-ll.args_v
		if np.round(ll.LL,8)<=np.round(LL0,8):#happens when the functions has not increased
			if k>10:
				print("Unable to reach convergence")
				return ll,g,G,H, 0 ,prtstr,constraints,dx_norm				

			k+=1
		else:
			k=0

		its+=1
		

	
	
	
def round_sign(x,n):
	"""rounds to n significant digits"""
	return round(x, -int(np.log10(abs(x)))+n-1)


def impose_OLS(ll,args_d,panel):
	beta,e=stat.OLS(panel,ll.X_st,ll.Y_st,return_e=True)
	args_d['omega'][0][0]=np.log(np.var(e*panel.included)*len(e[0])/np.sum(panel.included))
	args_d['beta'][:]=beta
	

def printout(_print,ll,dx_norm,panel,its,constraints,msg,window,H,G,CI):

	if not _print:
		return

	l=10
	pr=[['names','namelen',False,'Variable names',False,False],
	    ['args',l,True,'Coef',True,False],
	    ['direction',l,True,'direction',True,False],
	    ['se_robust',l,True,'SE(sandw.)',True,False],
	    ['sign_codes',5,False,'sign',False,False],
	    ['set_to',6,False,'set to',False,True],
	    ['assco',20,False,'associated variable',False,True],
	    ['cause',l,False,'cause',False,False]]	
	
	norm_prob,no_ac_prob=None,None
	if window.showNAC:
		ll.standardize()
		norm_prob=stat.JB_normality_test(ll.e_st,panel)		
		no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(panel,ll,10)
	
	o=output.output(pr, panel, H, G, 10, ll, 
	                             constraints,
	                             startspace='   ',
	                             direction=dx_norm)
	o.add_heading(its,
	              top_header=" "*75+"_"*12+"restricted variables"+"_"*13,
	              statistics=[['\nIndependent: ',panel.y_name[0],None,"\n"],
	                          ['Normality',norm_prob,3,'%'],
	                          ['P(no AC)',no_ac_prob,3,'%'],
	                          ['Max condition index',CI,3,'decimal']])
	o.add_footer(msg+'\n' + ll.errmsg)	
	o.print(window)
	return o.printstring


		
		
