#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tempfile
import numpy as np
import functions as fu
import pickle
import sys
import loaddata
import numpy.lib.recfunctions as rfn


if 'win' in sys.platform:
	fname=tempfile.gettempdir()+'\\paneltime.panel'
else:
	fname=tempfile.gettempdir()+'/paneltime.panel'
max_sessions=20

 
class args_bank:
	
	def __init__(self,X, Y, groups, W,loadargs):
		"""Load parameters if a similar model has been estimated before"""  
		self.session_db=load_obj()
		if (not loadargs) or (self.session_db is None):
			self.model_key=None
			(self.args,self.conv,self.not_in_use1,self.not_in_use2)=(None,0,None,None)
			return
		self.model_key=get_model_key(X, Y, groups, W)
		(d,a)=self.session_db
		if self.model_key in d.keys():
			(self.args,self.conv,self.not_in_use1,self.not_in_use2)=d[self.model_key]
		else:
			(self.args,self.conv,self.not_in_use1,self.not_in_use2)=(None,0,None,None)
	
	def save(self,args,conv,not_in_use1=None,not_in_use2=None):
		"""Saves the estimated parameters for later use"""
		f=open(fname, "w+b")
		if not self.session_db is None:
			d,a=self.session_db#d is d dictionary, and a is a sequental list that allows us to remove the oldest entry when the database is full
			if len(a)>max_sessions:
				d.pop(a.pop(),None)
		else:
			d=dict()
			a=[]
		d[self.model_key]=(args,conv,not_in_use1,not_in_use2)
		a.append(self.model_key)
		self.session_db=(d,a)
		pickle.dump((self.session_db),f)   
		f.flush() 
		f.close()



def load_obj():
	try:
		f=open(fname, "r+b")
	except:
		return None
	u= pickle.Unpickler(f)
	u=u.load()
	f.close()
	return u 


def get_model_key(X,Y, groups,W):
	"""Creates a string that is unique for the dataframe and arguments. Used to load starting values for regressions that
	have been run before"""
	s="%s%s%s%s" %(l(X),l(X**2),l(Y),l(Y**2))
	if not groups is None:
		s+="%s%s" %(l(groups),l(groups**2))
	if not W is None:
		s+="%s%s" %(l(W),l(W**2))
	return s

def l(x):
	n=5
	x=str(np.sum(x))
	if len(x)>n:
		x=x[len(x)-n:]
	return x


def parse_model(model_string):
	"""Parses model and returns the name of the Y variable and a list of the X variables as a tuple"""
	try:
		Y,X=model_string.split('~')
	except:
		raise RuntimeError("the 'model' argument should be a string on the form 'Y~X1 X2 X3' as separators for X-variables you can use either linefeed, space '+' or ','")
	Y=fu.clean(Y)
	for s in [',','\n','+',' ']:
		x_list=X.split(s)
		if len(x_list)>1:
			break
	if len(x_list)==1:
		X=fu.clean(x_list)
		return Y,X
	X=[]
	for i in x_list:
		x_name=fu.clean(i)
		if x_name!='':
			X.append(x_name)
	return Y,X

def test_dictionary(dataframe):
	n=0
	for i in dataframe.keys():
		if type(dataframe[i])==np.ndarray:
			if len(dataframe[i].shape)==1:
				dataframe[i]=dataframe[i].reshape((dataframe[i].shape[0],1))#converts to two dimensional
			if len(dataframe[i].shape)==2:
				if dataframe[i].shape[1]>1:
					raise RuntimeError("Variable %s has more than one column. This is not allowed." %(i,))		
			elif len(dataframe[i].shape)>2:
				raise RuntimeError("%s is a numpy ndarray with shape %s. Only one or two dimensional arrays are allowed" %(i,dataframe[i].shape))	
			if n==0:
				n=len(dataframe[i])
				s=i
			if len(dataframe[i])!=n:
				raise RuntimeError("Variables %s and %s have unequal number of observations. The number of observations must be identical for all variables. " %(i,s))	
			if n==0:
				raise RuntimeError("Variable %s has no observations. " %(i,s))	
	return n

def get_variables(dataframe,model_string,groups_name,w_names,add_intercept,sort_name):
	y_name,x_names=parse_model(model_string)
	groups,groups_name,void=check_var(dataframe,groups_name,'group_name')
	W,w_names,void=check_var(dataframe,w_names,'w_names',intercept_name='Ln(res.var), constant',raise_error=False,intercept_variable=True)
	intercept_name=None
	if add_intercept:
		intercept_name='Intercept'
	X,x_names,has_intercept=check_var(dataframe,x_names,'x_names',intercept_name=intercept_name,raise_error=True,intercept_variable=True)
	Y,y_names,void=check_var(dataframe,y_name,'y_name',raise_error=True)
	X,Y,groups,W=sort(dataframe,X,Y,groups,W,sort_name)
	return X,x_names,Y,y_name,groups,groups_name,W,w_names,has_intercept

def sort(dataframe,X,Y,groups,W,sort_name):
	if sort_name is None:
		return X,Y,groups,W
	if type(sort_name)==dict or type(sort_name)==np.ndarray:
		print ("Warning: 'sort_name' must be a string of the variable name in dataframe that shall be used for sorting")
		return X,Y,groups,W
	v=dataframe[sort_name]
	if not groups is None:
		dt=v.astype(dtype=[('date',type(v[0][0]))])
		g=groups.astype(dtype=[('groups',type(groups[0][0]))])
		v=rfn.merge_arrays((dt,g))
		srt=np.argsort(v,0,order=['groups','date']).flatten()
	else:
		srt=np.argsort(v,0).flatten()
	v=[X,Y,groups,W]
	for i in range(len(v)):
		if not v[i] is None:
			v[i]=v[i][srt]
	return v
	

def modify_dataframe(dataframe,transforms=None,filters=None):
	n=test_dictionary(dataframe)
	if 'ones' in dataframe.keys():
		print ("Warning: variable 'ones' is replaced by a constant vector of ones. Do not give any variable this name if you want to avoid this.")
	dataframe['ones']=np.ones((n,1))
	fu.exec_strip(transforms,globals(),dataframe)	
	n=filter_data(filters, dataframe,n)
	fu.exec_strip(transforms,globals(),dataframe)	
	print ("Checking and parsing variables ...")

def check_var(dataframe,names,arg_name,intercept_name=None,raise_error=False,intercept_variable=False):
	if names is None:
		if intercept_name is None:
			return None,None,None
		else:
			names=[]
	has_const=False
	if type(names)==str:
		names=[names]
	if type(names)!=list:
		raise RuntimeError("The %s argument must be a string or a list of strings" %(arg_name,))
	check_and_add_variables(names, dataframe, arg_name)
	if len(names)>0:
		has_const=np.all(dataframe[names[0]]==1)	
	if not intercept_name is None:
		if not has_const:
			names=[intercept_name]+names
		else:
			names[0]=intercept_name
		dataframe[intercept_name]=dataframe['ones']
		has_const=True
	X=[]
	for i in names:
		X.append(dataframe[i])
	X=np.concatenate(X,1)
	X,names=remove_constants(X,names, dataframe, raise_error,has_const)
	return X,names,has_const

	
def remove_constants(X,names,dataframe,raise_error,has_const):
	"""Removes constants variables at position 1 or higher, and any zero variable. You shold set raise_error=True for vital variables (X and Y)"""
	
	keep=np.var(X,0)!=0
	keep[0]=has_const or keep[0]#allways keep first variable unless all observations are aproximately zero
	sumtrash=np.sum(keep==0)
	if sumtrash>0:
		remvd=','.join(np.array(names)[keep==False])
		if sumtrash==1:
			remvd="Warning: The variable %s was removed because it was a constant or zero" %(remvd,)
		else:
			remvd="Warning: The variables %s were removed because they were constants or zero" %(remvd,)
		if raise_error and len(X[0])<=sumtrash:
			raise RuntimeError(remvd+'. Aborting since there are no more variables to run with.')
		else:
			print(remvd)
		if len(X[0])<=sumtrash:
			return None,None
	X=X[:,keep]
	names=list(np.array(names)[keep])
	return X,names

	

	
def check_and_add_variables(names,dataframe,arg_name):
	if names is None:
		return
	elif type(names)==str:
		names=[names]
	for name in names:
		if not name in dataframe:
			if (name.lower() in ['constant','ones','intercept','one','alpha']):
				dataframe[name]=dataframe['ones']
			else:
				try:
					var=eval(name,globals(),dataframe)
				except:
					raise RuntimeError("Variable %s is requested in %s, but it does not exist in the dataframe" %(name,arg_name))
				dataframe[name]=var
			
			

def filter_data(filters,dataframe,n):
	"""Filters the dataframe based on setting in the string Filters"""
	if filters is None:
		return n
	fltrs=fu.clean(filters,' and ')
	v=np.ones(n,dtype=bool)
	for f in fltrs:
		r=eval(f,globals(),dataframe)
		r.resize(n)
		print ('Removing %s observations due to filter %s' %(np.sum(r==0),f))
		v*=r
	for i in dataframe.keys():
		if type(dataframe[i])==np.ndarray:
			if len(dataframe[i])==n:
				dataframe[i]=dataframe[i][v]
				k=len(dataframe[i])
	print ('Removed %s of %s observations - %s observations remaining' %(n-k,n,k))
	return k



			
		
def get_data_and_model(X,Y,W=None,groups=None,x_names=None,y_name=None,w_names=None,groups_name=None,filters=None,transforms=None):
	"""Complies X and Y (and if supplied also W and groups) to a dataframe, and returns:\n
	- a dictionary with all variables """
	dataframe=dict()
	x_names=add_var_to_dict(dataframe, X, x_names, 'X')
	y_name=add_var_to_dict(dataframe, Y, y_name, 'Y')
	w_names=add_var_to_dict(dataframe, W, w_names, 'W')
	groups_name=add_var_to_dict(dataframe, groups, groups_name, 'groups')
	
	model_string="%s ~ %s" %(y_name[0], '+'.join(x_names))
	if not w_names is None:
		w_names=','.join(w_names)	
	modify_dataframe(dataframe,filters,transforms)
	return dataframe, model_string, w_names, groups_name
		


def add_var_to_dict(dataframe,V,v_names,arg_name):
	if V is None:
		if v_names is None:
			return None
		else:
			if type(v_names)==str:
				v_names=fu.clean(v_names,",")	
			return v_names
	if type(V)==dict:
		v_names=[]
		for i in V.keys():
			dim_check(V[i],arg_name,dict_check=True)
			dataframe[i]=V[i]
			v_names.append(i)
	elif type(V)==np.ndarray:
		v_names=dim_check(V,arg_name,v_names)
		for i in range(len(v_names)):
			dataframe[v_names[i]]=V[:,i:i+1]
	return v_names
			
			
			
		
def dim_check(V,arg_name,v_names=None,dict_check=False):
	if len(V.shape)!=2:
		raise RuntimeError ("%s is not two dimensional. Only two dimensional numpy_arrays are allowed" %(arg_name))	
	if dict_check:
		return
	N,k=V.shape
	if v_names is None:
		v_names=[arg_name+str(i) for i in range(k)]
	else:
		if type(v_names)==str:
			v_names=fu.clean(v_names,",")
		if len(v_names)!=k:
			raise RuntimeError ("A name list is supplied for %s, but its lenght does not match the number of columns in %s." %(arg_name,arg_name))	
	return v_names