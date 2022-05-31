#!/usr/bin/env python
# -*- coding: utf-8 -*-


import maximize_num


def maximize(panel,mp,args,channel,msg_main="",log=[]):
	"""Maxmizes logl.LL"""

	
	
	f,args,hessin,its,Convergence, ll = maximize_num.maximize(panel, args, step=1, 
										 callback='print', max_iter=10, gtol=0.01, tolx=0.1)	

	a = 0



		
def printout_func(percent,msg,ll,its,computation,incr,po,update_type, task=""):
	po.printout(ll,its+1,computation,incr,update_type)	
	return po.channel.set_progress(percent,msg,task=task)
		
class printout:
	def __init__(self,channel,panel,ll,computation,msg_main,_print=True,):
		self._print=_print
		self.channel=channel
		if channel is None:
			return
		channel.set_output_obj(ll, computation,msg_main)

		
		
	def printout(self,ll, its, computation,incr,update_type):
		if self.channel is None and self._print:
			print(ll.LL)
			return
		if not self._print:
			return
		if update_type>0:
			self.channel.update_after_direction(computation,its)
		elif update_type==0 or update_type==2:
			self.channel.update_after_linesearch(computation,ll,incr)

