#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import stat_functions as stat
import loglikelihood as logl
import output
from tkinter import _tkinter




def lnsrch(ll, direction,mp):
	rmsg=''
	args=ll.args_v
	g = direction.g
	dx = direction.dx
	panel=direction.panel
	LL0=ll.LL
	constr=direction.constr
	
	
	if np.sum(g*dx)<0:
		dx=-dx
		rmsg="convex function at evaluation point, direction reversed - "	
	if mp is None:
		return lnsrch_single(args, dx,panel,constr,rmsg)
	else:
		return lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0)
	

def lnsrch_single(args, dx,panel,constr,rmsg,f0=None):
	d=dict()
	if f0 is None:
		f0=logl.LL(args, panel,constr)
	d[f0.LL]=[f0,0]
	m=0.5
	f05,f1=None,None
	for i in range(16+len(dx)):#Setting lmda so that the largest step is valid. Set ll.LL to return None when input is invalid
		lmda=1.1*m**i #Always try full Newton step first.
		if i>14:
			dx=dx*(np.abs(dx)<max(np.abs(dx)))
		x=f0.args_v+lmda*dx
		f=logl.LL(x,panel,constr)
		if not f.LL is None:
			if f1 is None:
				f1,l1=f,lmda
			elif f05 is None:
				f05,l05=f,lmda
				break
	if f1 is None or f05 is None:
		return f0,'no valid values within newton step in linesearch',0,False
	d[f1.LL],d[f05.LL]=[f1,l1],[f05,l05]
	lambda_pred=solve_square_func(f0.LL,0,f05.LL,l05,f1.LL,l1)
	if not lambda_pred is None:
		f_pred=logl.LL(f0.args_v+lambda_pred*dx,panel,constr) 
		if not f_pred.LL is None:	
			d[f_pred.LL]=[f_pred,lambda_pred]
		lmda=min((lambda_pred,lmda))
	f_max=max(d.keys())
	if f_max>f0.LL:#the function has  increased
		return d[f_max][0],rmsg + 'Linesearch success',d[f_max][1],False
	for j in range(1,12):
		s=(0.5**j)
		lm=lmda*s
		ll=logl.LL(f0.args_v+lm*dx,panel,constr) 
		if not ll.LL is None:
			if ll.LL>f0.LL:
				return ll, rmsg+"Newton step in linesearch to big, found an increment at %s of Newton step" %(s,),lm,True

	return f0,rmsg+'No increase in linesearch',0,False
		
		
def solve_square_func(f0,l0,f05,l05,f1,l1,default=None):
	try:
		b=-f0*(l05+l1)/((l0-l05)*(l0-l1))
		b-=f05*(l0+l1)/((l05-l0)*(l05-l1))
		b-=f1*(l05+l0)/((l1-l05)*(l1-l0))
		c=((f0-f1)/(l0-l1)) + ((f05-f1)/(l1-l05))
		c=c/(l0-l05)
		if c<0 and b>0:#concave increasing function
			return -b/(2*c)
		else:
			return default
	except:
		return default
	
	
def lnsrch_master(args, dx,panel,constr,mp,rmsg,LL0):
	mp.send_dict_by_file({'constr':constr})
	start=0
	end=1.5
	msg='Linesearch success'
	newton_failed=False
	for i in range(4):
		delta=(end-start)/(mp.master.cpu_count-1)
		res=get_likelihoods(args, dx, panel, constr, mp,delta,start)
		if i==0:
			res0=res[0]		
		if (res[0,1]==0 and i==0) or (res[0,0]<=res0[0] and i>0) or np.isnan(res[0,0]):#best was no change
			start=delta/mp.master.cpu_count
			end=delta
		else:
			if i>0:
				msg=f'Found increment at {res[0,2]} of Newton step'
			break
	if i>0:
		res=np.append([res0],res,0)#ensuring the original is in the LL set
		srt=np.argsort(res[:,0])[::-1]
		res=res[srt]
	if LL0>res[0,0]:
		raise RuntimeWarning('Best linsearch is poorer than the starting point')
	try:
		lmda=solve_square_func(res[0,0], res[0,2],res[1,0], res[1,2],res[2,0], res[2,2],res[0,2])
		ll=logl.LL(args+lmda*dx,panel,constr)
		if ll.LL<res[0,0]:
			newton_failed=True
			raise RuntimeError('Somethong wrong with ll')
	except:
		ll, lmda = mp.remote_recieve(f'f{res[0,1]}',res[0,1]), res[0,2]
	if lmda==0:
		return ll,rmsg+'No increase in linesearch',0,True
	return ll,rmsg+msg,lmda,newton_failed

				
def get_likelihoods(args, dx,panel,constr,mp,delta,start):
	expr=[]	
	lamdas=[]
	args_lst=[]
	ids=range(mp.master.cpu_count)
	for i in ids:
		lmda=start+i*delta
		a=list(args+lmda*dx)
		lamdas.append(lmda)
		args_lst.append(a)
		expr.append([f"""
try:
	f{i}=lgl.LL({a}, panel,constr)
	LL{i}=f{i}.LL
except:
	LL{i}=None
""", f'LL{i}'])
	d=mp.remote_execute(expr)
	res=[]
	for i in ids:
		if not d[f'LL{i}'] is None:
			res.append([d[f'LL{i}'],i,lamdas[i]])
	if len(res)==0:
		return np.array([[np.nan, np.nan, np.nan]])
	res=np.array(res,dtype='object')
	srt=np.argsort(res[:,0])[::-1]
	res=res[srt]
	return res
	

	
	
def maximize(panel,direction,mp,args,window):
	"""Maxmizes logl.LL"""

	its, convergence_limit	= 0, 0.01
	k, m, dx_norm,incr		= 0,     0,    None, 0
	H,  digits_precision    = None, 12
	msg,lmbda,newton_failed	='',	1, False
	direction.hessin_num, ll= None, None
	args_archive			= panel.input.args_archive
	ll=direction.init_ll(args)
	po=printout(window,panel)
	po.printout(ll,'',its,direction,True,incr,lmbda)
	while 1:  	
		direction.get(ll,its,newton_failed)
		LL0=ll.LL
			
		#Convergence test:
		constr_conv=np.max(np.abs(direction.dx_norm))<convergence_limit
		unconstr_conv=np.max(np.abs(direction.dx_unconstr)) < convergence_limit 
		m=(m+1)*constr_conv
		if m>1 and (not unconstr_conv):
			newton_failed=not newton_failed
		conv=unconstr_conv or (m>1 and k>1)
		args_archive.save(ll.args_d,conv,panel.pqdkm)
		if conv:  #max direction smaller than convergence_limit -> covergence
			#if m==3:
			if _print: print("Convergence on zero gradient; maximum identified")
			po.printout(ll, "Convergence on zero gradient; maximum identified", its+1,direction,False,incr,lmbda)
			return ll,direction,po
			#m+=1
			#precise_hessian=precise_hessian==False

		
		po.printout(ll,'Linesearch', its+1,direction,False,incr,lmbda)
		ll,msg,lmbda,newton_failed=lnsrch(ll,direction,mp) 
		incr=ll.LL-LL0
		po.printout(ll, msg,its+1,direction,True,incr,lmbda)
		

		

		if round_sign(ll.LL,digits_precision)<=round_sign(LL0,digits_precision):#happens when the functions has not increased
			if k>10:
				print("Unable to reach convergence")
				return ll,direction,po			
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
	
class printout:
	def __init__(self,window,panel,_print=True):
		self._print=_print
		self.window=window
		if window is None:
			return
		self.panel=panel
		self.window.panel=panel
		self.window.settingsmenu.entryconfig(3,state='normal')	
		self.window.settingsmenu.entryconfig(4,state='normal')	
		
	def printout(self,ll,msg, its, direction, display_statistics,incr,lmbda):
		if self.window is None and self._print:
			print(ll.LL)
			return
		if not self._print:
			return
		self.window.ll=ll
		self.displaystats(display_statistics,ll)
				
		l=10
			#python variable name,	lenght,	not string,	 display name,	  		neg. values,	justification	col space
		pr=[['names',		'namelen',		False,	'Variable names',			False,		'right', 		''],
				['args',		l,			True,	'Coef',						True,		'right', 		'  '],
				['dx_norm',	    l,			True,	'direction',				True,		'right', 		'  '],
				['se_robust',	l,			True,	'SE(robust)',				True,		'right', 		'  '],
				['tstat',		l,			True,	't-stat.',					True,		'right', 		'  '],
				['tsign',		l,			True,	'significance',				False,		'right', 		'  '],
				['sign_codes',	5,			False,	'',						False,		'right', 		'  '],
				['assco',		20,			False,	'collinear with',			False,		'center', 		'  '],
				['set_to',		6,			False,	'value',					False,		'center', 		'  '],
				['cause',		l,			False,	'cause',					False,		'right', 		'  ']]			
		o=output.output(pr,ll, direction,self.panel.settings.robustcov_lags_run, startspace='   ')
		o.add_heading(its,
					  top_header=" "*118+"constraints",
					  statistics=[['\nIndependent: ',self.panel.input.y_name[0],None,"\n"],
								  ['Max condition index',direction.CI,3,'decimal']],
					  incr=incr)
		o.add_footer(msg+'\n' 
					 +f'Newton step: {lmbda}'
					 + ll.err_msg)	
		self.print(o)
		self.prtstr=o.printstring
		self.se_robust=o.se_robust
		
	def print(self,o):
		try:
			o.print(self.window)
		except Exception as e:
			test1=e.args[0]=="main thread is not in main loop"
			#test2=e==_tkinter.TclError
			if test1:
				exit(0)
			else:
				raise e		
		
	def displaystats(self,display_statistics,ll):
		if display_statistics:
			self.window.right_tabs.process_charts.initialize()
			if self.window.right_tabs.process_charts.btn_stats.cget('relief')=='sunken':
				norm_prob=stat.JB_normality_test(ll.e_st_centered,self.panel)		
				no_ac_prob,rhos,RSqAC=stat.breusch_godfrey_test(self.panel,ll,10)
				norm_prob,no_ac_prob=np.round([norm_prob,no_ac_prob],5)*100			
				self.window.right_tabs.process_charts.statistics.set(f"""
  Normality:\t{norm_prob} %\n
  Stationarity:\t{no_ac_prob} %
				""")
				self.window.right_tabs.process_charts.plot(ll)	



		
		
