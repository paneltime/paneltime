#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
class options_item:
	def __init__(self,value,description,dtype,name,permissible_values=None,value_description=None, descr_for_vector_setting='',category='General'):
		"""permissible values can be a vector or a string with an inequality, 
		where %s represents the number, for example "1>%s>0"\n
		if permissible_values is a vector, value_description is a corresponding vector with 
		description of each value in permissible_values"""
		#if permissible_values
		self.description=description
		self.value=value
		self.dtype=dtype
		if type(dtype)==str:
			self.dtype_str=dtype
		elif type(dtype)==list or type(dtype)==tuple:
			self.dtype_str=str(dtype).replace('<class ','').replace('[','').replace(']','').replace('>','').replace("'",'')
		else:
			self.dtype_str=dtype.__name__
			
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
			return False
		if i is None:
			if self.value!=value:
				self.value=value
		else:
			if self.value[i]!=value:
				self.value[i]=value
			else:
				return False
		return True
	
	def valid(self,value,i=None):
		if self.permissible_values is None:
			return (type(value)==self.dtype) or (type(value) in self.dtype)		
		if i is None:
			return self.valid_test(value, self.permissible_values)
		else:
			return self.valid_test(value, self.permissible_values[i])
			
	def valid_test(self,value,permissible):
		if type(permissible)==list or type(permissible)==tuple:
			try:
				if not type(value)==list or type(value)==tuple:
					value=self.dtype(value)
					return value in permissible
				else:
					valid=True
					for i in range(len(value)):
						value[i]=self.dtype(value[i])
						valid=valid*eval(permissible[i] %(value[i],))
			except:
				return False
			return valid
		elif type(permissible)==str:
			return eval(permissible %(value,))
		else:
			print('No method to handle this permissible')
		

		
class options():
	def __init__(self):
		pass
		
		
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



def regression_options():
	#Add option here for it to apear in the "options"-tab. The options are bound
	#to the data sets loaded. Hence, a change in the options here only has effect
	#ON DATA SETS LOADED AFTER THE CHANGE
	self=options()
	self.pqdkm						= options_item([1,1,0,1,1], 
																	["Auto Regression order (ARIMA, p)",
																	"Moving Average order (ARIMA, q)",
																	"difference order (ARIMA, d)",
																	"Variance Moving Average order (GARCH, k)",
																	"Variance Auto Regression order (GARCH, m)"],int, 'ARIMA-GARCH orders',
																	["%s>=0","%s>=0","%s>=0","%s>=0","%s>=0"],
																	descr_for_vector_setting="ARIMA-GARCH parameters",category='Regression')
	self.group_fixed_random_eff		= options_item(2,				'Fixed, random or no group effects', int, 'Group fixed random effect',[0,1,2], 
																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	self.time_fixed_random_eff		= options_item(2,				'Fixed, random or no time effects', int, 'Time fixed random effect',[0,1,2], 
																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	self.variance_fixed_random_eff	= options_item(2,				'Fixed, random or no group effects for variance', int, 'Variance fixed random effects',[0,1,2], 
																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	
	self.loadargs					= options_item(1, 				"Determines whether the regression arguments from the previous run should be kept", 
																	int, 'Load arguments', [0,1,2],
																	['No loading',
																	'Load from last run of identical model',
																	'Load from last run'
																	])
	
	self.arguments					= options_item("", 				arg_desc, str, 'Initial arguments', descr_for_vector_setting="Arguments (dictionary):")	
	
	
	self.loadARIMA_GARCH			= options_item(False, 				"Determines whether the ARIMA_GARCH arguments from the previous run should be kept", 
																	bool, 'Load ARIMA_GARCH', [False,True],
																	['No loading',
																	'Load arguments'
																	])	
	
	self.add_intercept				= options_item(True,			"If True, adds intercept if not all ready in the data",
																	bool,'Add intercept', [True,False],['Add intercept','Do not add intercept'],category='Regression')
	
	self.h_function					= options_item(h_func,			h_descr, str,"GARCH function",category='Regression')
	self.user_constraints			= options_item(None,			constr_str,str, 'User constraints')
	
	self.tobit_limits				= options_item([None,None],		['lower limit','upper limit'], [float,type(None)], 'Tobit-model limits', descr_for_vector_setting=tobit_desc)
	
	self.min_group_df				= options_item(1, 				"The smallest permissible number of observations in each group. Must be at least 1", int, 'Minimum degrees of freedom', "%s>0",category='Regression')
	self.robustcov_lags_statistics	= options_item([100,30],		[robust_desc_0,robust_desc_1], int, 'Robust covariance lags (time)', ["%s>1","%s>1"], descr_for_vector_setting=robust_desc_all,category='Output')
	self.silent						= options_item(False, 			silent_desc,  bool,'Silent mode',[True,False],['Silent','Not Silent'])
	#self.description				= options_item(None, 			descr_descr, 'entry','Description')	
	
	self.convergence_limit			= options_item([0.0001], ['Convergence limit:'],float,"Convergence limit", ["%s>0"],
										descr_for_vector_setting=conv_desc
										)	
	
	self.do_not_constraint			= options_item([None], ['Variable not to constraint:'],[str,type(None)],"Avoid constraint",
										descr_for_vector_setting="The name of a variable of interest \nthat shall not be constrained due to \nmulticollinearity"
										)	
	
	self.minimum_iterations	= options_item(0, 'Minimum number of iterations in maximization:',int,"Minimum iterations", "%s>-1")		
	
	self.make_category_tree()
	
	return self


def application_preferences():
	opt=options()
	
	opt.save_datasets	= options_item(True, "If True, all loaded datasets are saved on exit and will reappear when the application is restarted", 
									bool,"Save datasets on exit", [False,True],
									['Save on exit',
									'No thanks'])
	
	opt.n_round	= options_item(4, "Sets the number of digits the results are rounded to", 
									str,"Rounding digits", ['no rounding','0 digits','1 digits','2 digits','3 digits',
																						 '4 digits','5 digits','6 digits','7 digits','8 digits',
																						 '9 digits','10 digits'])
	
	opt.n_digits	= options_item(10, "Sets the maximum number of digits (for scientific format) if 'Rounding digits' is not set (-1)", 
									int,"Number of digits", ['0 digits','1 digits','2 digits','3 digits',
																						 '4 digits','5 digits','6 digits','7 digits','8 digits',
																						 '9 digits','10 digits'])	
	opt.round_scientific	= options_item(True, "Determines if small numbers that are displayed in scientific format shall be rounded", 
								   bool,"Round Scientific", [True,False],['Round Scientific','Do not round scientific'])		
	opt.make_category_tree()
	
	return opt


h_func="""
def h(e,z):
	e2			=	e**2+1e-5
	h_val		=	np.log(e2)	
	h_e_val		=	2*e/e2
	h_2e_val	=	2/e2-4*e**2/e2**2

	return h_val,h_e_val,h_2e_val,None,None,None
"""

conv_desc="""
Convergence limit. When this exceedes the maximum absolute value of 
new directions (as percentage of parameter value), the procedure is 
determined to have converged"""
h_descr="""
You can supply your own heteroskedasticity function. It must be a function of\n
residuals e and a shift parameter z that is determined by the maximization procedure\n
the function must return the value and its derivatives in the following order:\n
h, dh/de, (d^2)h/de^2, dh/dz, (d^2)h/dz^2,(d^2)h/(dz*de)"""

constr_str="""
You can add constraints in python dictonary syntax.\n
"""

tobit_desc="""
Determines the limits in a tobit regression. 
Element 0 is lower limit and element1 is upper limit. 
If None, the limit is not active"""

robust_desc_all="""
Numer of lags used in calculation of the robust \ncovariance matrix for the time dimension.\n"""
robust_desc_0="""# lags in final statistics calulation       """
robust_desc_1="""# lags iterations (smaller saves time) """

silent_desc="""
True if silent mode, otherwise False. 
For running the procedure in a script, where output should be suppressed"""

descr_descr="""
A description of the project. 
Used in the final output and to load the project later.\n
default is the model_string"""

arg_desc="""A string with a dictionary in python syntax, 
containing the initial arguments. 
An example can be obtained by
printing ll.args.args_d"""

