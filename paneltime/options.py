#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
class options_item:
	def __init__(self,value,description,dtype,name,permissible_values=None,value_description=None, descr_for_input_boxes=[],category='General'):
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
		self.descr_for_input_boxes=descr_for_input_boxes
		self.category=category
		self.name=name
		self.selection_var= len(descr_for_input_boxes)==0 and type(permissible_values)==list
		self.is_inputlist=len(self.descr_for_input_boxes)>0
	
		
		
	def set(self,value,i=None):
		try:
			if not self.valid(value,i):
				return False
		except Exception as e:
			a=self.valid(value,i)
			print(e)
			return False
		if i is None:
			if str(self.value)!=str(value):
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
	
	self.add_intercept				= options_item(True,			"If True, adds intercept if not all ready in the data",
																	bool,'Add intercept', [True,False],['Add intercept','Do not add intercept'],category='Regression')

	self.ARMA_constraint	        = options_item(1.0,				'Maximum absolute value of ARMA coefficients', float, 'ARMA coefficient constraint',
																	None,None,category='GARCH')	
		
	
	self.arguments					= options_item("", 				"A string or dict with a dictionary in python syntax containing the initial arguments." 
																	"An example can be obtained by printing ll.args.args_d"
																	, [str,dict], 'Initial arguments')	

	#self.description				= options_item(None, 			"A description of the project." , 'entry','Description')	
	self.normal_GARCH	            = options_item(True,			'Normal GARCH, as opposed to EGARCH if True', bool, 'Estimate GARCH directly',
																	[True,False],['Direct GARCH','Usual GARCH'],category='GARCH')	
		
	
	self.do_not_constrain			= options_item(None, 			"The name of a variable of interest \nthat shall not be constrained due to \nmulticollinearity",
													 				[str,type(None)],"Avoid constraint",
																	descr_for_input_boxes=['Variable not to constraint:'])	
	
	self.fixed_random_group_eff		= options_item(2,				'Fixed, random or no group effects', int, 'Group fixed random effect',[0,1,2], 
																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	self.fixed_random_time_eff		= options_item(2,				'Fixed, random or no time effects', int, 'Time fixed random effect',[0,1,2], 
																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	self.fixed_random_variance_eff	= options_item(0,				'Fixed, random or no group effects for variance', int, 'Variance fixed random effects',[0,1,2], 
																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	
	
	self.RE_in_GARCH      = options_item(False,			'GARCH process is calculated after FE/RE', bool, 'If True, GARCH is calcualted after FE/RE, an not ',
																	[True,False],['GARCH calcualted after FE/RE','GARCH calcualted before FE/RE'],category='Fixed-random effects')	
	
	self.h_function					= options_item(					"def h(e,z):\n"
																	"	e2			=	e**2+1e-5\n"
																	"	h_val		=	np.log(e2)\n"	
																	"	h_e_val		=	2*e/e2\n"
																	"	h_2e_val	=	2/e2-4*e**2/e2**2\n"
																	"	return h_val,h_e_val,h_2e_val,None,None,None\n",	
													
																	"You can supply your own heteroskedasticity function. It must be a function of\n"
																	"residuals e and a shift parameter z that is determined by the maximization procedure\n"
																	"the function must return the value and its derivatives in the following order:\n"
																	"h, dh/de, (d^2)h/de^2, dh/dz, (d^2)h/dz^2,(d^2)h/(dz*de)"
																	, str,"GARCH function",category='Regression')

	self.kurtosis_adj				= options_item(0,				'Amount of kurtosis adjustment', float, 'Amount of kurtosis adjustment',
																	"%s>=0",category='GARCH')	
	
	self.GARCH_assist				= options_item(0,				'Amount of weight put on assisting GARCH variance to be close to squared residuals', float, 'GARCH assist',
																	"%s>=0",category='GARCH')		
		
	self.loadargs					= options_item(1, 				"Determines whether the regression arguments from the previous run should be kept", 
																	int, 'Load arguments', [0,1,2],
																	['No loading',
																	'Load from last run of identical model',
																	'Load from last run'
																	])
	
	self.min_group_df				= options_item(1, 				"The smallest permissible number of observations in each group. Must be at least 1", int, 'Minimum degrees of freedom', "%s>0",category='Regression')

	self.minimum_iterations			= options_item(0, 				'Minimum number of iterations in maximization:',
													  				int,"Minimum iterations", "%s>-1")		
	
	#self.pedantic					= options_item(0, 				"Determines how pedantic the maximization shuould be\n"
	#																"If slightly pedantic, the linesearch is allways run twice after the direction is calcualted\n"
	#																"If pedantic, the maximization also tries to fix supected variables\n"
	#																"If very pedantic the maximization also uses brute force as a last resort",  
	#																int,'Pedantic',[0,1,2,3],['Not pedantic', 'Slightly pedantic','Pedantic','Very pedantic'])
	
	self.pool						= options_item(False, 			"True if sample is to be pooled, otherwise False." 
																	"For running a pooled regression",  
																	bool,'Pooling',[True,False],['Pooled','Not Pooled'])
	
	self.pqdkm						= options_item([1,1,0,1,1], 
																	"ARIMA-GARCH parameters:",int, 'ARIMA-GARCH orders',
																	["%s>=0","%s>=0","%s>=0","%s>=0","%s>=0"],
																	descr_for_input_boxes=["Auto Regression order (ARIMA, p)",
																	"Moving Average order (ARIMA, q)",
																	"difference order (ARIMA, d)",
																	"Variance Moving Average order (GARCH, k)",
																	"Variance Auto Regression order (GARCH, m)"],category='Regression')

	self.robustcov_lags_statistics	= options_item([100,30],		"Numer of lags used in calculation of the robust \ncovariance matrix for the time dimension", 
																	 int, 'Robust covariance lags (time)', ["%s>1","%s>1"], 
													 	 	 	 	 descr_for_input_boxes=["# lags in final statistics calulation",
																	 "# lags iterations (smaller saves time)"],
																	 category='Output')

	self.silent						= options_item(False, 			"True if silent mode, otherwise False." 
																	"For running the procedure in a script, where output should be suppressed",  
																	bool,'Silent mode',[True,False],['Silent','Not Silent'])

	self.subtract_means				= options_item(False,			"If True, subtracts the mean of all variables. This may be a remedy for multicollinearity if the mean is not of interest.",
																	bool,'Subtract means', [True,False],['Subtracts the means','Do not subtract the means'],category='Regression')
	
	self.tobit_limits				= options_item([None,None],		"Determines the limits in a tobit regression. "
																	"Element 0 is lower limit and element1 is upper limit. "
																	"If None, the limit is not active", 
																	[float,type(None)], 'Tobit-model limits', 
																	descr_for_input_boxes=['lower limit','upper limit'])

	self.tolerance					= options_item(0.000001, 		"Tolerance. When the maximum absolute value of the gradient divided by the hessian diagonal"
																	"is smaller than the tolerance, the procedure is "
																	"determined to have converged.",
																	float,"Tolerance", "%s>0")	
	
	self.variance_RE_norm			= options_item(0.000005, 		"This parameter determines at which point the log function involved in the variance RE/FE calculations, "
																	"will be extrapolate by a linear function for smaller values",
																	float,"Variance RE/FE normalization point in log function", "%s>0")		
	self.user_constraints			= options_item(None,			"You can add constraints as a dict or as a string in python dictonary syntax.\n",
																	[str,dict], 'User constraints')
	
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


