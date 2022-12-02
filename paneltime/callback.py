#!/usr/bin/env python
# -*- coding: utf-8 -*-

#This module handle callbacks

QUIT_EXCEPTION = 'Quitting on demand'

class CallBack:
	def __init__(self, channel = None):
		self.channel = channel
		self.inbox = {}
		self.outbox = {}
		self.terminate = False
		
	def set_communication(self, comm, _print=True):
		self.comm = comm
		self._print = _print
		
	def callback(self, **keywordargs):
		for k in keywordargs:
			self.outbox[k] = keywordargs[k]
		
	def callin(self, **keywordargs):
		for k in keywordargs:
			self.inbox[k] = keywordargs[k]		

	def print(self, msg, its, incr, ll, perc , task, dx_norm):
		if not self._print or self.channel is None:
			return
		if not self.channel.output_set:
			self.channel.set_output_obj(ll, self.comm, dx_norm)
		self.channel.set_progress(perc ,msg ,task=task)
		self.channel.update(self.comm,its,ll,incr, dx_norm)
		a=0
	
	def print_final(self, msg, its, incr, fret, perc, task, conv, dx_norm, t0, xsol, ll, node):
		self.print(msg, its, incr, ll, perc, task, dx_norm)
		self.channel.print_final(msg, fret, conv, t0, xsol, its, node)
		a=0



	
	
	