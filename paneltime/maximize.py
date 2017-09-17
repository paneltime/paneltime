#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import statproc as stat
import loglikelihood as logl
import os
import time

digits_precision=7

def lnsrch(f0, g, dx,panel):

	s=g*dx
	slope=np.sum(s)					#ensuring positive slope (should not be negative unless errors in gradient and/or hessian)
	if slope<=0:
		print("Warning: Roundoff problem in lnsrch")
		return f0
	m=0.25
	for i in range(15+len(dx)):#Setting lmda so that the largest step is valid. Set ll.LL to return None when input is invalid
		lmda=m**i #Always try full Newton step first.
		if i>14:
			dx=dx*(np.abs(dx)<max(np.abs(dx)))
		x=f0.args_v+lmda*dx
		f1=logl.LL(x,panel)
		if not f1.LL is None:
			break
	if i==14+len(x):
		return f0
	i=0
	d={f0.LL:f0,f1.LL:f1}

	while 1:
		i+=1
		f05=logl.LL(f0.args_v+lmda*(0.5**i)*dx,panel)
		if not f05.LL is None:		
			break
	d[f05.LL]=f05
	for i in []:
		fcheck=logl.LL(f0.args_v+lmda*i*dx,panel)
		if not fcheck.LL is None:
			d[fcheck.LL]=fcheck
	b=-(4*(f0.LL-f05.LL)+(f1.LL-f0.LL))/lmda
	c=2*((f0.LL-f05.LL)+(f1.LL-f05.LL))/(lmda**2)
	lambda_pred=lmda*0.25
	if c<0 and b>0:#concave increasing function
		lambda_pred=-b/(2*c)
		f_pred=logl.LL(f0.args_v+lambda_pred*dx,panel) 
		if not f_pred.LL is None:	
			d[f_pred.LL]=f_pred
	
	f_max=max(d.keys())
	if f_max==f0.LL:#the function has not increased
		for j in range(1,6):
			lmda=lambda_pred*(0.05**j)
			ll=logl.LL(f0.args_v+lmda*dx,panel) 
			if ll.LL is None:
				break
			if ll.LL>f0.LL:
				return ll
	else:
		return d[f_max]
	return f0#should never happen	
		
		
def maximize(panel,direction,mp,direction_testing,args_archive,args=None,_print=True,user_constraints=None):
	"""Maxmizes logl.LL"""
	
	ll=logl.LL(args,panel)
	if ll.LL is None:
		print("""You requested stored arguments from a previous session 
		to be used as initial arguments (loadargs=True) but these failed to 
		return a valid log likelihood with the new parameters. Default inital 
		arguments will be used. """)
		ll=logl.LL(panel.args.args,panel)
	its=0
	mc_limit_init=300
	mc_limit_min=0
	mc_limit=mc_limit_init
	convergence_limit=0.01
	k=0
	dx_conv=None
	H=None
	dxi=None
	g=None
	if direction_testing:
		t=time.time()
		ll=dirtest(ll, panel,direction,mp,user_constraints)
		print(time.time()-t)
	direction.hessin_num=None
	while 1:  
		its+=1
		dx,g,G,H,constrained,reset=direction.get(ll,mc_limit,dx_conv,k,its,mp,dxi=dxi,user_constraints=user_constraints)
		if reset:
			k=0
		f0=ll
		LL0=round_sign(ll.LL,digits_precision)
		dx_conv=(ll.args_v!=0)*np.abs(dx)*(constrained==0)/(np.abs(ll.args_v)+(ll.args_v==0))
		dx_conv=(ll.args_v==0)*dx+dx_conv
		printout(_print, ll, dx_conv,panel,its)
		#Convergence test:
		if np.max(dx_conv) < convergence_limit and (its>3 or  np.sum(constrained)<=2):  #max direction smaller than convergence_limit -> covergence
			if _print: print("Convergence on zero gradient; maximum identified")
			return ll,g,G,H,1
		ll=lnsrch(ll,g,dx,panel) 

		args_archive.save(ll.args_d,0)
		
		dxi=f0.args_v-ll.args_v
		test=np.max(np.abs(f0.args_v-ll.args_v)/np.maximum(np.abs(ll.args_v),1e-50))
		if round_sign(ll.LL,digits_precision)==LL0 or (test < 12.0e-16 ):#happens when the functions has not increased or arguments not changed
			if np.sum(constrained)>=len(constrained):
				print("Unable to reach convergence")
				return ll,g,G,H, 0 
			if mc_limit==mc_limit_min:
				mc_limit=mc_limit_init#increases the number of restricted variables
			else:
				mc_limit=mc_limit_min
				k+=1
		else:
			mc_limit=mc_limit_init
			k=0
	

def round_sign(x,n):
	"""rounds to n significant digits"""
	return round(x, -int(np.log10(abs(x)))+n-1)


def dirtest(ll,panel,direction,mp,user_constraints):
	#dx,dx_approx,g,G,H,constrained,reset=direction.get(ll,1000,None,0,0,mp)
	#ll=lnsrch(ll,g,dx,panel)
	ll.standardize()
	if os.cpu_count()<24:
		ll=pretest_sub(ll,['psi','gamma'],panel,direction,mp,user_constraints)
		ll=pretest_sub(ll,['rho','lambda'],panel,direction,mp,user_constraints)
		
	else:
		ll=pretest_master(ll,panel,direction,mp,user_constraints)
	return ll
	
	
def pretest_master(ll,panel,direction,mp,user_constraints):
	c=['psi','gamma','rho','lambda']
	k=len(c)
	a=np.array(np.meshgrid(*tuple([[-0.5, 0.5]]*k))).T.reshape(-1,k)
	n=len(a)
	args=[]
	for i in a:
		args_d=ll.copy_args_d()
		for j in range(k):
			args_d[c[j]][0]=i[j]
		args.append(args_d)
	expr=[]
	n_cores=os.cpu_count()
	expr=['' for i in range(n_cores)]
	i=0
	while 1:
		for j in range(n_cores):
			s="t0=time.time()\n"
			s+="tmp=pretest_func(panel,direction,args[%s],ll,user_constraints)\n" %(i,)
			expr[j]+=s+'res%s=tmp,(time.time()-t0),time.time(),t0\n'  %(i,)

			i+=1
			if i==n:
				break
		if i==n:
			break			
	d={'args':args,'ll':ll,'user_constraints':user_constraints,
	              'pretest_func':pretest_func,'lnsrch':lnsrch,'panel':panel,
	              'direction':direction}

	mp.send_dict(d,'dynamic dictionary')	
	d=mp.execute(expr)
	
	max_ll=ll
	for i in range(n):
		ll_new,dt,t1,t0=d['res%s' %(i,)]
		print('pretest LL: %s  Max LL: %s, dt: %s,t1: %s, t0: %s' %(ll_new.LL,max_ll.LL,dt,t1,t0))
		if not ll_new is None:
			if ll_new.LL>max_ll.LL:
				max_ll=ll_new	
	return max_ll
	
def pretest_func(panel,direction,args,ll,user_constraints,mp=None):
	
	ll_new=logl.LL(args,panel)
	
	if ll_new.LL is None:
		return ll
	try:
		dx,g,G,H,constrained,reset=direction.get(ll_new,1000,None,0,-1,mp)
	except:
		return ll	
	ll_new=lnsrch(ll_new,g,dx,panel) 
	
	return ll_new
	
def pretest_sub(ll,categories,panel,direction,mp,user_constraints):
	c=categories
	vals=[-0.5,0,0.5]
	max_ll=ll
	args_d=ll.copy_args_d()
	for i in vals:
		for j in vals:
			#if np.sign(i)==np.sign(i) or np.sign(i)>0:
			#	break
			for k in [0,1]:
				if len(args_d[c[k]])>0:
					args_d[c[k]][0]=[i,j][k]
			#impose_OLS(ll,args_d, panel)
			ll_new=pretest_func(panel, direction, args_d, ll,user_constraints,mp)
			catstr="(%s,%s)" %tuple(c)
			print('Direction test %s=(%s,%s) LL: %s  Max LL: %s' %(catstr,i,j,ll_new.LL,max_ll.LL))
			if ll_new.LL>max_ll.LL:
				max_ll=ll_new
	return max_ll

def impose_OLS(ll,args_d,panel):
	beta,e=stat.OLS(panel,ll.X_st,ll.Y_st,return_e=True)
	args_d['omega'][0][0]=np.log(np.var(e*panel.included)*len(e[0])/np.sum(panel.included))
	args_d['beta'][:]=beta
	

def printout(_print,ll,dx_conv,panel,its):
	ll.standardize()
	norm_prob=stat.JB_normality_test(ll.e_st,panel)	
	if _print: 
		print("LL: %s Normality probability: %s    Iteration: %s" %(ll.LL,norm_prob,its))
		a='['
		k=0
		for i in np.round(dx_conv*100,2):
			k+=1
			if k==6:
				k=0
				a=a+(str(i)+ '00')[0:4].rjust(9) + ' %,\n'
			elif k==1 and a=='[':
				a=a+(str(i)+ '00')[0:4].rjust(9) + ' %,'
			elif k==1:
				a=a+(str(i)+ '00')[0:4].rjust(10) + ' %,'
			else:
				a=a+(str(i)+ '00')[0:4].rjust(9) + ' %,'
		print("New direction as percentage of argument: \n%s" %(a[:-1]+']',))
		print("Coefficients : \n%s" %(ll.args_v,))

		
		
