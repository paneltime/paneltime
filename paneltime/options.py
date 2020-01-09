#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
class options_item:
	def __init__(self,value,description,dtype,name,permissible_values=None,value_description=None,descr_for_vector_setting='',category='General'):
		"""permissible values can be a vector or a string with an inequality, 
		where %s represents the number, for example "1>%s>0"\n
		if permissible_values is a vector, value_description is a corresponding vector with 
		description of each value in permissible_values"""
		#if permissible_values
		self.description=description
		self.value=value
		self.dtype=dtype
		self.permissible_values=permissible_values
		self.value_description=value_description
		self.descr_for_vector_setting=descr_for_vector_setting
		self.category=category
		self.name=name
		self.selection_var= (descr_for_vector_setting=='') and type(permissible_values)==list
		self.is_inputlist=len(self.descr_for_vector_setting)>0 and type(self.description)==list
	
		
		
	def set(self,value,i=None):
		try:
			if not self.valid(value,i):
				return False
		except Exception as e:
			return e
		self.value=value
		return True
	
	def valid(self,value,i=None):
		if i is None:
			return valid(value, self.permissible_values)
		else:
			return valid(value, self.permissible_values[i])
			

		
class options:
	def __init__(self):
		self.pqdkm						= options_item([1,1,0,1,1], 
								 										["Auto Regression order (ARIMA, p)",
								  										"Moving Average order (ARIMA, q)",
																		"difference order (ARIMA, d)",
								  										"Variance Moving Average order (GARCH, k)",
																		"Variance Auto Regression order (GARCH, m)"],int, 'ARIMA-GARCH orders',
								 										["%s>=0","%s>=0","%s>=0","%s>=0","%s>=0"],
								 										descr_for_vector_setting="ARIMA-GARCH parameters",category='Regression')
		self.group_fixed_random_eff		= options_item(2,				'Fixed, random or no group effects', str, 'Group fixed random effect',[0,1,2], 
																		['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
		self.time_fixed_random_eff		= options_item(2,				'Fixed, random or no time effects', str, 'Time fixed random effect',[0,1,2], 
																		['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
		self.variance_fixed_random_eff	= options_item(2,				'Fixed, random or no group effects for variance', str, 'Variance fixed random effects',[0,1,2], 
																		['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
		
		self.loadargs					= options_item(1, 				"Determines whether the arguments from the previous iteration should be kept", 
																		int, 'Load arguments', [0,1,2],
																		['No loading of arguments',
														 				'Load arguments',
														 				'Load arguments and prevous ARIMA/GARCH-order'])
		
		self.add_intercept				= options_item(True,			"If True, adds intercept if not all ready in the data",
														  				bool,'Add intercept', [True,False],['Add intercept','Do not add intercept'],category='Regression')
		
		self.h_function					= options_item(h_func,			h_descr, str,"GARCH function",category='Regression')
		self.user_constraints			= options_item(None,			constr_str,str, 'User constraints')
		
		self.tobit_limits				= options_item([None,None],		['lower limit','upper limit'], float, 'Tobit-model limits', descr_for_vector_setting=tobit_desc)
		
		self.min_group_df				= options_item(5, 				"The smallest permissible degrees of freedom", int, 'Minimum degrees of freedom', "%s>0",category='Regression')
		self.robustcov_lags_statistics	= options_item([100,30],		[robust_desc_0,robust_desc_1], int, 'Robust covariance lags (time)', ["%s>1","%s>1"], descr_for_vector_setting=robust_desc_all,category='Output')
		self.silent						= options_item(False, 			silent_desc,  bool,'Silent mode',[True,False],['Silent','Not Silent'])
		self.description				= options_item(None, 			descr_descr, 'entry','Description')
		
		self.make_category_tree()
		
		
	def make_category_tree(self):
		opt=self.__dict__
		d=dict()
		keys=np.array(list(opt.keys()))
		keys=keys[keys.argsort()]
		for i in opt:
			if opt[i].category in d:
				d[opt[i].category].append(opt[i])
			else:
				d[opt[i].category]=[opt[i]]
			opt[i].code_name=i
		self.categories=d	
		keys=np.array(list(d.keys()))
		self.categories_srtd=keys[keys.argsort()]





h_func="""
def h(e,z):
	e2			=	e**2+1e-5
	h_val		=	np.log(e2)	
	h_e_val		=	2*e/e2
	h_2e_val	=	2/e2-4*e**2/e2**2

	return h_val,h_e_val,h_2e_val,None,None,None
"""

h_descr="""
You can supply your own heteroskedasticity function. It must be a function of\n
residuals e and a shift parameter z that is determined by the maximization procedure\n
the function must return the value and its derivatives in the following order:\n
h,\n
dh/de,(d^2)h/de^2,\n
dh/dz,(d^2)h/dz^2\n
(d^2)h/(dz*de)\n"""

constr_str="""
You can add constraints in python dictonary syntax.\n
"""

tobit_desc="""
Determines the limits in a tobit regression. 
Element 0 is lower limit and element1 is upper limit. 
If None, the limit is not active"""

robust_desc_all="""
Numer of lags used in calculation of the robust covariance matrix for the time dimension."""
robust_desc_0="""Number of lags in final statistics calulation"""
robust_desc_1="""Number of lags used each iteration (smaller value saves time)"""

silent_desc="""
True if silent mode, otherwise False. 
For running the procedure in a script, where output should be suppressed"""

descr_descr="""
A description of the project. 
Used in the final output and to load the project later.\n
default is the model_string"""


def valid(value,permissible):
	if type(permissible)==list or type(permissible)==tuple:
		return value in permissible
	elif type(permissible)==str:
		return eval(permissible %(value,))
	else:
		raise RuntimeError('No method to handle this permissible')