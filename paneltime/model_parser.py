#!/usr/bin/env python
# -*- coding: utf-8 -*-
DEFAULT_INTERCEPT_NAME='Intercept'
import numpy as np
import functions as fu
import loaddata
import numpy.lib.recfunctions as rfn



def parse_model(model_string):
	"""Parses model and returns the name of the Y variable and a list of the X variables as a tuple"""
	try:
		Y,X=model_string.split('~')
	except:
		try:
			Y,X=model_string.split('=')
		except:
			raise RuntimeError("the 'model' argument should be a string on the form 'Y~X1 X2 X3' or 'Y=X1 X2 X3'. As separators for X-variables you can use either linefeed, space '+' or ','")
	Y=fu.clean(Y)
	X=fu.clean(X).replace('++','+')
	X=X.split('+')
	for i in range(len(X)):
		X[i]=fu.clean(X[i])
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
				raise RuntimeError("Variable %s has no observations. " %(i))	
	return n


def get_variables(input_class,dataframe,model_string,IDs_name,time_name,settings):
	print ("Analyzing variables ...")
	for i in dataframe:
		if type(dataframe[i])==np.ndarray:
			if len(dataframe[i])==0:
				raise RuntimeError("There are no valid observations")
			else:
				break
			
	sort(dataframe,time_name,IDs_name)
	
	input_class.timevar,input_class.time_name,void=check_var(dataframe,time_name,'time_name')
	input_class.IDs,input_class.IDs_name,void=check_var(dataframe,IDs_name,'ID_name')
	input_class.lag_obj=lag_object(input_class.IDs)
	dataframe['L']=input_class.lag_obj.lag#allowing for lag operator in model input
	dataframe['D']=input_class.lag_obj.diff#allowing for lag operator in model input
	input_class.W,input_class.W_names,void=check_var(dataframe,settings.heteroscedasticity_factors,'heteroscedasticity_factors',
                                       intercept_name='log variance constant',raise_error=False)
	input_class.Z,input_class.z_names,void=check_var(dataframe,settings.instruments,'instruments',intercept_name='instrument intercept',raise_error=False)	
	intercept_name=None
	if settings.add_intercept.value>0:
		intercept_name=DEFAULT_INTERCEPT_NAME
	trend_name=None
	if settings.add_intercept.value==2:
		trend_name=DEFAULT_INTERCEPT_NAME	
	(input_class.X,input_class.x_names,
	 input_class.has_intercept,
	 input_class.Y,input_class.y_name)=parse_and_check(dataframe,model_string,intercept_name,trend_name,input_class)

		
def parse_and_check(dataframe,model_string,intercept_name,trend_name,input_class):
	y_name,x_names=parse_model(model_string)
	if x_names==['']:
		x_names=['intercept']
	X,x_names,has_intercept=check_var(dataframe,x_names,'x_names',intercept_name=intercept_name,raise_error=True)
	Y,y_name,void=check_var(dataframe,y_name,'y_name',raise_error=True)	
	return X,x_names,has_intercept,Y,y_name


class lag_object:

	def __init__(self,IDs):
		self.IDs=IDs
		self.max_lags=0
		
	def lag(self,variable,lags=1):
		v=np.roll(variable, lags)
		idroll=np.roll(self.IDs,lags)
		keep=idroll==self.IDs
		self.max_lags=max((self.max_lags,lags))
		return v*keep
	
	def diff(self,variable,lags):
		return variable-self.lag(variable, lags)
		

def sort(dataframe,time_name,IDs_name):
	"sorts the dataset"
	if (time_name is None) and (IDs_name is None):
		return
	elif (time_name is None) and (not IDs_name is None):
		g=dataframe[IDs_name]
		if np.var(g)==0:
			return
		srt=np.argsort(g,0).flatten()
	elif (not time_name is None) and (IDs_name is None):
		dt=dataframe[time_name]
		if np.var(dt)==None:
			return
		srt=np.argsort(dt,0).flatten()	
	else:
		dt=dataframe[time_name]
		vdt=np.var(dt)
		dt=dt.astype(dtype=[('date',type(dt[0][0]))])
		g=dataframe[IDs_name]
		vg=np.var(g)
		g=g.astype(dtype=[('IDs',type(g[0][0]))])
		if vdt==0 and vg==0:
			return
		if vdt==0 and vg>0:
			srt=np.argsort(g,0).flatten()
		elif vdt>0 and vg==0:
			srt=np.argsort(dt,0).flatten()
		else:
			s=rfn.merge_arrays((dt,g))
			srt=np.argsort(s,0,order=['IDs','date']).flatten()
	for i in dataframe:
		if type(dataframe[i])==np.ndarray:
			dataframe[i]=dataframe[i][srt]

	

def modify_dataframe(dataframe,transforms=None,filters=None):
	print ("Checking and parsing variables ...")
	n=test_dictionary(dataframe)
	if 'ones' in dataframe.keys():
		print ("Warning: variable 'ones' is replaced by a constant vector of ones. Do not give any variable this name if you want to avoid this.")
	dataframe['ones']=np.ones((n,1))
	if not transforms is None:
		exec(transforms,globals(),dataframe)	
	if not filters is None:
		n,d=filter_data(filters, dataframe)
		if not transforms is None:
			exec(transforms,globals(),dataframe)
	for i in list(dataframe.keys()):
		if callable(dataframe[i]):
			dataframe.pop(i)		
	print ("... done")

def check_var(dataframe,names,arg_name,intercept_name=None,raise_error=False):
	names=parse_names(names, intercept_name, arg_name)
	if names is None: return None,None,None
	check_and_add_variables(names, dataframe, arg_name)
	has_const,names=check_intercept(names, dataframe, intercept_name)
	X=[]
	for i in names:
		X.append(dataframe[i])
	X=np.concatenate(X,1)
	X,names=remove(X,names, dataframe, raise_error,has_const)
	return X,names,has_const
	

def check_intercept(names,dataframe,intercept_name):
	"The intercept MUST be the FIRST variable if supplied as a variable"
	has_const=False
	if len(names)>0:
		has_const=np.all(dataframe[names[0]]==1)
	if has_const:
		return True,names
	if not (intercept_name is None):
		if (intercept_name in names):
			if (not has_const):#the user has all ready suplied the intercept
				raise RuntimeError(f"If an intercept variable named '{intercept_name}' is supplied by the user, it needs to be the first variable and a constant equal to one. ")
		elif not has_const:
			names=[intercept_name]+names
			dataframe[intercept_name]=np.array(dataframe['ones'])
		has_const=True
	return has_const,names

def parse_names(names,intercept_name,arg_name):
	if names is None:
		if intercept_name is None:
			return None
		else:
			names=[]
	if type(names)==str:
		names=names.split(',')
	if type(names)!=list:
		raise RuntimeError("The %s argument must be a string or a list of strings" %(arg_name,))	
	return names
	
def remove(X,names,dataframe,raise_error,has_const):
	"""Removes constants variables at position 1 or higher, and any zero variable. You shold set raise_error=True for vital variables (X and Y)"""
	
	keep=np.var(X,0)!=0
	keep[0]=has_const or keep[0]#allways keep first variable unless all observations are aproximately zero
	sumtrash=np.sum(keep==0)
	if sumtrash>0:
		remvd=','.join(np.array(names)[keep==False])
		if sumtrash==1:
			remvd="Warning: The variable %s was removed because it was constant" %(remvd,)
		else:
			remvd="Warning: The variables %s were removed because they were constant" %(remvd,)
		if raise_error and len(X[0])<=sumtrash:
			raise RuntimeError(remvd+'. Aborting since there are no more variables to run with.')
		else:
			a=0
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
		if (name.lower() in ['constant','ones','intercept','one','alpha']):
			dataframe[name]=dataframe['ones']
		else:
			try:
				var=eval(name,globals(),dataframe)
			except KeyError as e:
				raise RuntimeError("Variable %s is requested in %s, but it does not exist in the dataframe" %(name,arg_name))
			dataframe[name]=var
			
			

def filter_data(filters,dataframe,copy=True):
	"""Filters the dataframe based on setting in the string Filters"""
	if filters is None:
		return None,None
	if not ' and ' in filters:
		if type(filters)==str:
			fltrs=filters.split()
		else:
			fltrs=filters
	else:
		fltrs=fu.clean(filters,' and ')
	n=len(dataframe[list(dataframe.keys())[0]])
	v=np.ones(n,dtype=bool)
	removed=0
	filter_results={}
	for f in fltrs:
		r=eval(f,globals(),dataframe)
		if type(r)==tuple:
			raise RuntimeError("Filters must be entered as a list of strings")
		r.resize(n)
		removed=np.sum((r==0)*(v==1))
		if not copy:print ('Removing %s observations due to filter %s' %(removed,f))
		filter_results[f]=removed
		v*=r
	if copy:
		d=filtered_dict(filter_results, n)
	else:
		d=dataframe
	for i in dataframe:
		if type(dataframe[i])==np.ndarray:
			if len(dataframe[i])==n:
				d[i]=dataframe[i][v]
				k=len(d[i])
	if not copy:print ('Removed %s of %s observations - %s observations remaining' %(n-k,n,k))
	return d,k

class filtered_dict(dict):
	def __init__(self,filter_results,filter_original_sample_size,iterable=[]):
		dict.__init__(self,iterable)
		self.filter_results=filter_results
		self.filter_original_sample_size=filter_original_sample_size

			
		
def get_data_and_model(X,Y,W=None,IDs=None,x_names=None,y_name=None,heteroscedasticity_factors=None,IDs_name=None,filters=None,transforms=None):
	"""Complies X and Y (and if supplied also W and IDs) to a dataframe, and returns:\n
	- a dictionary with all variables """
	dataframe=dict()
	x_names=add_var_to_dict(dataframe, X, x_names, 'X')
	y_name=add_var_to_dict(dataframe, Y, y_name, 'Y')
	heteroscedasticity_factors=add_var_to_dict(dataframe, W, heteroscedasticity_factors, 'W')
	IDs_name=add_var_to_dict(dataframe, IDs, IDs_name, 'IDs')
	
	model_string="%s ~ %s" %(y_name[0], '+'.join(x_names))
	if not heteroscedasticity_factors is None:
		heteroscedasticity_factors=','.join(heteroscedasticity_factors)	
	modify_dataframe(dataframe,filters,transforms)
	return dataframe, model_string, heteroscedasticity_factors, IDs_name
		


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


def check_sign(results,stats,category,oldp,fixed,maxp=1000,sign_level=0.05):
	"Checks whether the higest order of category is significant. If it is not, lim is set to True"
	sign=stats.output.tsign[results.panel.args.positions[category]]
	if len(sign)==0:
		return 1,fixed
	members=results.panel.args.positions[category]
	a=list(results.constraints.fixed)
	b=list(results.constraints.associates)
	c=np.unique(a+b)
	n=len(sign)
	sign=sign[::-1]
	for i in range(n):
		constrained=np.any([j in c for j in members])
		if np.all(sign[i:])<sign_level and not constrained:
			if n-i<oldp or n-i+1>maxp:
				return n-i,True
			else:
				return n-i+1,False
	return 1,fixed

