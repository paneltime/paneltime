#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
def create_options():
	options = options_dict()
	options_to_txt(options)
	opt = OptionsObj(options)
	return opt

def options_to_txt(options):
	with open('options.txt','w') as f:
		for o in options:
			if type(options[o].dtype) is list:
				tp = [i.__name__ for i in options[o].dtype]
			else:
				tp = options[o].dtype.__name__
			f.write(f"|{o}|{options[o].value}|{tp}|{options[o].name}:{options[o].description.replace('\n', '')}|\n")

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



	def set(self,value):
		try:
			if not self.valid(value):
				return False
		except Exception as e:
			a=self.valid(value)
			print(e)
			return False
		if str(self.value)!=str(value):
			self.value=value
		return True

	def valid(self,value):
		if self.permissible_values is None:
			try:
				if self.dtype(value)==value:
					return True
			except:
				pass
			if type(value) in self.dtype:
				return True
		return self.valid_test(value, self.permissible_values)


	def valid_test(self,value,permissible):
		if permissible is None:
			return True
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

class OptionsObj:
	def __init__(self, options):
		for o in options:
			super().__setattr__('_' + o, options[o]) 
			super().__setattr__(o, options[o].value) 

		self.make_category_tree()

	def __setattr__(self, name, value):
		# Trigger a custom function when an attribute is set
		if '_' + name in self.__dict__:
			self._options[name].set(value)
			value = self._options[name].value
		super().__setattr__(name, value)  # Perform the actual attribute assignment

	def make_category_tree(self):
		opt=self.__dict__
		d=dict()
		keys=np.array(list(opt.keys()))
		keys=keys[keys.argsort()]
		for i in opt:
			if i[0]=='_':
				if opt[i].category in d:
					d[opt[i].category].append(opt[i])
				else:
					d[opt[i].category]=[opt[i]]
				opt[i].code_name=i
		self.categories=d	
		keys=np.array(list(d.keys()))
		self.categories_srtd=keys[keys.argsort()]


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



def options_dict():
	#Add option here for it to apear in the "options"-tab. The options are bound
	#to the data sets loaded. Hence, a change in the options here only has effect
	#ON DATA SETS LOADED AFTER THE CHANGE
	options = {}
	options['accuracy']					= options_item(0, 				"Accuracy of the optimization algorithm. 0 = fast and inaccurate, 3=slow and maximum accuracy", int, 
																																						'Accuracy', "%s>0",category='Regression')

	options['add_intercept']					= options_item(True,			"If True, adds intercept if not all ready in the data",
																																				 bool,'Add intercept', [True,False],['Add intercept','Do not add intercept'],category='Regression')
	
	options['arguments']						= options_item(None, 				"A dict or string defining a dictionary in python syntax containing the initial arguments." 
																																						 "An example can be obtained by printing ll.args.args_d"
																																																																				, [str,dict, list, np.ndarray], 'Initial arguments')	

	options['ARMA_constraint']		        = options_item(1000,				'Maximum absolute value of ARMA coefficients', float, 'ARMA coefficient constraint',
																													 None,None,category='ARIMA-GARCH')	
	options['betaconstraint']		          = options_item(True,			'Determines whether to initially constraint beta coefficients while setting the ARIMA-GARCH-coefficients', bool, 
																															'Constraint betas initially',
																															[True,False],['Constraint betas','Do not constraint betas'],category='Regression')	

	options['constraints_engine']		        = options_item(True,			'Determines whether to use the constraints engine', bool, 'Uses constraints engine',
																															[True,False],['Use constraints','Do not use constraints'],category='Regression')	

	options['debug_mode']		      	        = options_item(False,			'Determines whether the code will run in debug mode. Should normally allways be False', 
																																		bool, 'Debug or not',
																																		 [True,False],['Debug mode','Not debug mode'],category='General')	

	options['multicoll_threshold_report']	 = options_item(30,				'Threshold for reporting multicoll problems', float, 'Multicollinearity threshold',
																											 None,None)		

	options['multicoll_threshold_max']	    = options_item(1000,			'Threshold for imposing constraints on collineary variables', float, 'Multicollinearity threshold',
																											 None,None)			

	#options['description']					= options_item(None, 			"A description of the project." , 'entry','Description')	
	options['EGARCH']		            = options_item(False,			'Normal GARCH, as opposed to EGARCH if True', bool, 'Estimate GARCH directly',
																											[True,False],['Direct GARCH','Usual GARCH'],category='ARIMA-GARCH')	


	options['do_not_constrain']				= options_item(None, 			"The name of a variable of interest \nthat shall not be constrained due to \nmulticollinearity",
																																		[str,type(None)],"Avoid constraint",
																																		descr_for_input_boxes=['Variable not to constraint:'])	

	options['fixed_random_group_eff']			= options_item(0,				'Fixed, random or no group effects', int, 'Group fixed random effect',[0,1,2], 
																																	['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	options['fixed_random_time_eff']			= options_item(0,				'Fixed, random or no time effects', int, 'Time fixed random effect',[0,1,2], 
																																 ['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	options['fixed_random_variance_eff']		= options_item(0,				'Fixed, random or no group effects for variance', int, 'Variance fixed random effects',[0,1,2], 
																														 ['No effects','Fixed effects','Random effects'],category='Fixed-random effects')



	options['h_function']						= options_item("def h(e,z):\n"
																				"	e2			=	e**2+1e-5\n"
																													"	h_val		=	np.log(e2)\n"	
																													"	h_e_val		=	2*e/e2\n"
																													"	h_2e_val	=	2/e2-4*e**2/e2**2\n"
																													"	return h_val,h_e_val,h_2e_val,None,None,None\n",	

																													"You can supply your own heteroskedasticity function. It must be a function of\n"
																													"residuals e and a shift parameter z that is determined by the maximization procedure\n"
																													"the function must return the value and its computation in the following order:\n"
																													"h, dh/de, (d^2)h/de^2, dh/dz, (d^2)h/dz^2,(d^2)h/(dz*de)"
																													, str,"GARCH function",category='Regression')
	options['include_initvar']					= options_item(True,			"If True, includes an initaial variance term",
																																				 bool,'Include initial variance', [True,False],['Include','Do not include'],category='Regression')

	options['initial_arima_garch_params']	 = options_item(0.1,			'The initial size of arima-garch parameters (all directions will be attempted', 
																											 float, 'initial size of arima-garch parameters',
																																																																									 "%s>=0",category='ARIMA-GARCH')		

	options['kurtosis_adj']					= options_item(0,				'Amount of kurtosis adjustment', float, 'Amount of kurtosis adjustment',
																																				"%s>=0",category='ARIMA-GARCH')	

	options['GARCH_assist']					= options_item(0,				'Amount of weight put on assisting GARCH variance to be close to squared residuals', float, 'GARCH assist',
																																				"%s>=0",category='ARIMA-GARCH')		

	options['min_group_df']					= options_item(1, 				"The smallest permissible number of observations in each group. Must be at least 1", int, 'Minimum degrees of freedom', "%s>0",category='Regression')

	options['max_iterations']				= options_item(150, 			"Maximum number of iterations", int, 'Maximum number of iterations', "%s>0",category='Regression')
	
	options['max_increments']				= options_item(0, 			"Maximum increment before maximization is ended", float, 'Maximum increments', "%s>0",category='Regression')

	options['minimum_iterations']			= options_item(0, 				'Minimum number of iterations in maximization:',
																																			int,"Minimum iterations", "%s>-1")		


	options['pool']							= options_item(False, 			"True if sample is to be pooled, otherwise False." 
																									"For running a pooled regression",  
																									bool,'Pooling',[True,False],['Pooled','Not Pooled'])

	options['pqdkm']							= options_item([1,1,0,1,1], 
															"ARIMA-GARCH parameters:",int, 'ARIMA-GARCH orders',
																"%s>=0",
																descr_for_input_boxes=["Auto Regression order (ARIMA, p)",
																												"Moving Average order (ARIMA, q)",
																"difference order (ARIMA, d)",
																"Variance Moving Average order (GARCH, k)",
																"Variance Auto Regression order (GARCH, m)"],category='Regression')

	options['robustcov_lags_statistics']		= options_item([100,30],		"Numer of lags used in calculation of the robust \ncovariance matrix for the time dimension", 
																														 int, 'Robust covariance lags (time)', "%s>1", 
																																descr_for_input_boxes=["# lags in final statistics calulation",
																																													"# lags iterations (smaller saves time)"],
																																		category='Output')

	options['silent']							= options_item(False, 			"True if silent mode, otherwise False." 
																																									"For running the procedure in a script, where output should be suppressed",  
																											bool,'Silent mode',[True,False],['Silent','Not Silent'])

	options['subtract_means']					= options_item(False,			"If True, subtracts the mean of all variables. This may be a remedy for multicollinearity if the mean is not of interest.",
																											 bool,'Subtract means', [True,False],['Subtracts the means','Do not subtract the means'],category='Regression')

	options['supress_output']					= options_item(True,			"If True, no output is printed.",
																												 bool,'Supress output', [True,False],['Supresses output','Do not supress output'],category='Regression')

	options['tobit_limits']					= options_item([None,None],		"Determines the limits in a tobit regression. Element 0 is lower limit and element1 is upper limit. "
																										"If None, the limit is not active", 
																										[float,type(None)], 'Tobit-model limits', 
																										descr_for_input_boxes=['lower limit','upper limit'])

	options['tolerance']						= options_item(0.001, 		"Tolerance. When the maximum absolute value of the gradient divided by the hessian diagonal"
																																						 "is smaller than the tolerance, the procedure is "
																														"Tolerance in maximum likelihood",
																														float,"Tolerance", "%s>0")	
	
	options['ARMA_round']						= options_item(14, 		"Number og digits to round elements in the ARMA matrices by. Small differences in these values can "
																								"change the optimization path and makes the estimate less robust"
																											"Number of significant digits in ARMA",
																											int,"# of signficant digits", "%s>0")	  

	options['variance_RE_norm']				= options_item(0.000005, 		"This parameter determines at which point the log function involved in the variance RE/FE calculations, "
																																		"will be extrapolate by a linear function for smaller values",
																											 float,"Variance RE/FE normalization point in log function", "%s>0")		
	options['user_constraints']				= options_item(None,			"You can add constraints as a dict or as a string in python dictonary syntax.\n",
																																		[str,dict], 'User constraints')

	options['use_analytical']			= options_item(1,				'Use analytical Hessian', int, 'Analytical Hessian',[0,1,2], 
																																	['No analytical','Analytical in some iterations','Analytical in all iterations'],category='Genereal')


	options['web_open_tab']					= options_item(True, 			"True if web a new web browser tab should be opened when using web interface" 
																																				"Should a new tab be opemed?",  
																									bool,'New web tab',[True,False],['Yes','No'])	


	return options


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


