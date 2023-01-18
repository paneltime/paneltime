#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module handle callbacks

import time

CALLBACK_INTERVAL = 1

class CallBack:
	def __init__(self, qin, qout, callback_active):
		self.qin = qin
		self.qout = qout
		self.inbox = {}
		self.outbox = {'quit':False}
		self.t = time.time()
		self.callback_active = callback_active
		
	def callback(self, **keywordargs):
		for k in keywordargs:
			self.outbox[k] = keywordargs[k]
		if not self.callback_active:
			return		
		if time.time()-self.t<CALLBACK_INTERVAL:
			return
		self.t = time.time()
		msgin = {}
		if not self.qin.empty():
			msgin = self.qin.get()
		inbox = dict(self.inbox)
		for k in msgin:
			inbox[k] = msgin[k]
		self.inbox = inbox
		self.qout.put(self.outbox)
		print('called back')

		
			

			




	
	
	