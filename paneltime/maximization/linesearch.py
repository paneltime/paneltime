#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .. import likelihood as logl

from . import direction
import numpy as np


STPMX=100.0 
import time
class LineSearch:
	def __init__(self, x, comput, panel, ll_old, step = 1):
		self.alf = 1.0e-3     #Ensures sufficient decrease in function value.
		self.tolx = 1.0e-14  #Convergence criterion on fx.		
		self.step = step
		self.stpmax = STPMX * max((abs(np.sum(x**2)	))**0.5,len(x))
		self.comput = comput
		self.panel = panel
		self.applied_constraints = []
		self.ll_old = ll_old

	def lnsrch(self, x, f, g, H, dx):

		#(x, f, g, dx) 

		self.conv=0
		self.rev = False
		self.g = g
		self.msg = ""
		n=len(x)
	
		if f is None:
			raise RuntimeError('f cannot be None')

		summ=np.sum(dx**2)**0.5
		if summ > self.stpmax:
			dx = dx*self.stpmax/summ 
		if np.all(dx==0):
			self.default(f, x, 0,  "dx is zero", 4)  
			return      
			
		test=0.0 															#Compute lambda min.
		for i in range(0,n): 
			temp=abs(dx[i])/max(abs(x[i]),1.0) 
			if (temp > test): test=temp 
		alamin = self.tolx/test 
		#*******CUSTOMIZATION
		#multithread:

		for i in range(20):#Setting alam so that the largest step is valid. Set func to return None when input is invalid
			self.alam = 0.5**i**2*self.step #Always try full Newton step first.
			dx_alam, slope, self.rev, self.applied_constraints  = direction.new(g, x, H, self.comput.constr, f, dx, self.alam)
			self.x = x + dx_alam
			self.f, self.ll = self.func(self.x) 
			if self.f != None: 
				if self.f>=f:
					break
			

		if self.f is None:
			self.default(f, x, 0,  "dx is zero", 4)  
			return 
		#*************************
		f2=f
		f0 = f
		ll0 = self.ll
		alam2 = self.alam
		alamstart = self.alam#***********CUSTOMIZATION
		max_iter = 500
		for self.k in range (0,max_iter):			#Start of iteration loop.
			dx_alam, slope, self.rev, self.applied_constraints  = direction.new(g, x, H, self.comput.constr, f, dx, self.alam)
			self.x = x + dx_alam
			if self.k > 0: 
				self.f, self.ll = self.func(self.x) 
			if self.f is None:
				self.msg = 'Function reached undefined region'
				self.f = f0
				self.ll = ll0
				return
			if (self.alam < alamin):   #Convergence on delta x. For zero finding,the calling program should verify the convergence.
				self.default(f, x, 0, "Convergence on delta dx", 3)
				return
			elif (self.f >= f+self.alf*self.alam*slope): 
				self.msg = "Sufficient function increase"
				self.conv = 1
				self.step_adj()
				return							#Sufficient function increase
			else:  															#Backtrack.
				if (self.alam == alamstart):#***********CUSTOMIZATION  alam == 1.0
					tmplam = -slope/(2.0*(self.f-f-slope))  	#First time.
				else:  														#Subsequent backtracks.
					rhs1 = self.f-f-self.alam*slope 
					rhs2 = f2-f-alam2*slope 
					a=(rhs1/(self.alam**2)-rhs2/(alam2*alam2))/(self.alam-alam2) 
					b=(-alam2*rhs1/(self.alam**2)+self.alam*rhs2/(alam2*alam2))/(self.alam-alam2) 
					if (a == 0.0):
						tmplam = -slope/(2.0*b)  
					else:  
						disc=b*b-3.0*a*slope 
						if (disc < 0.0):
							tmplam = 0.5*self.alam  
						elif (b >= 0.0):
							tmplam=-(b+(disc)**0.5)/(3.0*a) 
						else:
							tmplam=slope/(-b+(disc)**0.5)
					if (tmplam > 0.5*self.alam): 
						tmplam = 0.5*self.alam   								#  lambda<=0.5*lambda1
			alam2 = self.alam 
			if self.f>f0:
				f0 = self.f
				ll0 = self.ll
			f2 = self.f
			self.alam = max(tmplam, 0.1*self.alam)								#lambda>=0.1*lambda1
			if alamstart<1.0:#*************CUSTOMIZATION
				self.alam = min((self.alam, alamstart*0.9**self.k))
			
		self.msg = f"No function increase after {max_iter} iterations"
		self.conv = 2

	def step_adj(self):
		if self.alam==self.step:
			self.step += self.step
		elif self.step>1:
			if self.step>2.0:
				self.step = 0.5*self.step
			else:
				self.step=1.0

	def func(self,x):	
		ll = logl.LL(x, self.panel, self.comput.constr)
		if ll is None:
			return None, None
		elif ll.LL is None:
			return None, None
		return ll.LL, ll

	def default(self, f, x, alam, msg, conv):
		self.msg = msg
		self.conv = conv
		self.f = f
		self.x = x
		self.alam = alam
		self.ll = self.ll_old
