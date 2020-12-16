#contains a numpy.ndarray derived class that manages zero observations in a panel

import numpy as np

class array(np.ndarray):
	def __new__(cls, array,included,ignore=False):
		obj = np.asarray(array).view(cls)
		cls.included=np.array(included)
		cls.ignore=ignore
		return obj

	
	def __add__(self,other):
		res=empty_handling(self,other)
		if not res is None:
			return res
		res=np.add(self,other)
		res=np.multiply(res,self.included[3])
		return res
		
	def __sub__(self,other):
		res=empty_handling(self,other)
		if not res is None:
			return res		
		res=np.subtract(self,other)
		res=np.multiply(res,self.included[3])
		return res		
		
	def __mul__(self,other):
		res=empty_handling(self,other)
		if not res is None:
			return res		
		res=np.multiply(self,other)
		res=np.multiply(res,self.included[3])
		return res		
		
	def __truediv__(self,other):
		res=empty_handling(self,other)
		if not res is None:
			return res		
		res=np.true_divide(self,other)
		res=np.multiply(res,self.included[3])
		return res
		
	def __eq__(self,other):
		res=empty_handling(self,other)
		if not res is None:
			return res		
		a=np.multiply(self,self.included[3])
		b=np.multiply(other,self.included[3])		
		res=np.equal(a,b)
		return res		


def empty_handling(self,other):
	if self.ignore:
		if len(other)==0:
			return self
		elif len(self)==0:
			return other
		else:
			return None
	else:
		if len(self)==0 or len(other)==0:
			return self[len(self):]	
		else:
			return None
		
a=array([2,4],[0,1])+array([8,10],[0,1])
f=a+np.array([])

a=0
