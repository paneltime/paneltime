#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import pickle
import tempfile
import os


fname_args=os.path.join(tempfile.gettempdir(),'paneltime.args')
fname_data=os.path.join(tempfile.gettempdir(),'paneltime.data')
fname_key=os.path.join(tempfile.gettempdir(),'paneltime.key')
fname_temprec=os.path.join(tempfile.gettempdir(),'paneltime.temprec')
fname_window=os.path.join(tempfile.gettempdir(),'paneltime.win')
max_sessions=20

class args_archive:

	def __init__(self,model_string,loadargs):
		"""Loads parameters if a similar model has been estimated before. Stored parameters are accessed by
		creating an instance of this class"""  
		
		self.model_key=model_string# possibility for determening model based on data: get_model_key(X, Y, W)
		self.data=load_obj(fname_args)
		if (not loadargs) or (self.data is None):
			(self.args,self.conv,self.pqdkm,self.not_in_use2)=(None,0,(0,0,0,0,0),None)
			return
		(d,a)=self.data
		if self.model_key in d.keys():
			(self.args,self.conv,self.pqdkm,self.not_in_use2)=d[self.model_key]
		else:
			(self.args,self.conv,self.pqdkm,self.not_in_use2)=(None,0,(0,0,0,0,0),None)

	def load(self):#for debugging
		data=load_obj(fname_args)
		(d,a)=data
		if self.model_key in d.keys():
			return d[self.model_key]
		else:
			return (None,0,None,None)		

	def save(self,args,conv,pqdkm,not_in_use2=None):
		"""Saves the estimated parameters for later use"""
		if not self.data is None:
			d,a=self.data#d is d dictionary, and a is a sequental list that allows us to remove the oldest entry when the database is full
			if (len(a)>max_sessions) and (not self.model_key in d):
				d.pop(a.pop(0))
		else:
			d=dict()
			a=[]
		d[self.model_key]=(args,conv,pqdkm,not_in_use2)
		if self.model_key in a:
			a.remove(self.model_key)
		a.append(self.model_key)		
		if len(a)!=len(d):
			a=list(d.keys())
		self.data=(d,a)
		save_obj(fname_args,self.data)



def loaddata(key):
	"""Loads data if a similar data was loaded before. """  
	
	key=key.replace('\n',' ').replace('\r',' ').replace('\t',' ').replace('  ',' ').replace('  ',' ')
	current_key=load_obj(fname_key)
	if key==current_key:
		return load_obj(fname_data)
	
	
def savedata(key,data):
	"""saves data  """
	key=key.replace('\n',' ').replace('\r',' ').replace('\t',' ').replace('  ',' ').replace('  ',' ')
	save_obj(fname_key,key)
	save_obj(fname_data,data)
	

def load_model(self):#for debugging
	data=load_obj(datafname)
	(d,a)=data
	if self.model_key in d.keys():
		return d[self.model_key]
	else:
		return (None,0,None,None)		

def load_obj(fname):
	try:
		f=open(fname, "r+b")
		u= pickle.Unpickler(f)
		u=u.load()
		f.close()
		return u 
	except:
		return None
	
	
def save_obj(fname,obj):
	f=open(fname, "w+b")
	pickle.dump(obj,f)   
	f.flush() 
	f.close()	

def get_model_key(X,Y, IDs,W):
	"""Creates a string that is unique for the dataframe and arguments. Used to load starting values for regressions that
	have been run before"""
	s="%s%s%s%s" %(l(X),l(X**2),l(Y),l(Y**2))
	if not IDs is None:
		s+="%s%s" %(l(IDs),l(IDs**2))
	if not W is None:
		s+="%s%s" %(l(W),l(W**2))
	return s

def l(x):
	n=5
	x=str(np.sum(x))
	if len(x)>n:
		x=x[len(x)-n:]
	return x


class tempfile_manager:
	def __init__(self):
		#at initialization, all temporary files from previous sesstions are deleted
		self.rec=load_obj(fname_temprec)
		if self.rec is None:
			self.rec=[]
		for f in self.rec:
			try:
				os.remove(f)
			except:
				pass
		save_obj(fname_temprec,[])
		
	def TemporaryFile(self):
		f=tempfile.TemporaryFile()
		self.rec.append(f.name)
		save_obj(fname_temprec,self.rec)
		return f
	
	