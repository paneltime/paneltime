#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os

def create_options():
	options = options_dict()
	opt = OptionsObj(options)
	return opt

def options_to_txt():

	options = options_dict()
	a = []

	for o in options:
		opt = options[o]
		if type(opt.dtype) is list:
			tp = [i.__name__ for i in opt.dtype]
		else:
			tp = opt.dtype.__name__
		value = opt.value
		if type(value)==str:
			value = value.replace('\n','<br>').replace('\t','a&#9;')
			if len(value)>12:
				value = value[:9]+"..."
		perm = opt.permissible_values
		if perm == None:
			perm = 'Any'
		a.append([o, value, tp, opt.permissible_values , f"<b>{opt.name}:</b> {opt.description}".replace('\n','<br>').replace('\t','a&#9;')])

	sorted_list = sorted(a, key=lambda x: x[0])
	path = os.sep.join(__file__.split(os.sep)[:-2])
	with open(f'{path}{os.sep}qmd{os.sep}options.qmd','w') as f:
		f.write(	"---\n"
					"title: Setting options\n"
					"nav_order: 2\n"
					"has_toc: true\n"
					"---\n\n\n"
					"# Setting options\n\n\n"
					
					"You can set various options by setting attributes of the `options` attribute, for example:\n"
					"```\n"
					"import paneltime as pt\n"
					"pt.options.accuracy = 1e-10\n"
					"```\n\n"
					"## `OptionsObj` attributes \n\n\n"
					"|Attribute name|Default<br>value|Permissible<br>values*|Data<br>type|Description|\n"
					"|--------------|-------------|-----------|-----------|-----------|\n")
		
		for name, default, dtype, perm, desc in sorted_list:
			f.write(f"|{name}|{default}|{perm}|{dtype}|{desc}|\n")
	a=0

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
			self.dtype_str= 'NA'

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
			if self.dtype is type:
				isclass = isinstance(value, self.dtype)
				if not isclass:
					raise TypeError(f"Expected type 'type' (class type) for option {self.name} but got {type(value)}")
				return True
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
			if type(value) == list or type(value)== tuple:
				return np.all([eval(permissible %(i,)) for i in value])
			else:
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
		_name = '_' + name
		if _name in self.__dict__:
			try:
				self.__dict__[_name].set(value)
			except Exception as e:
				raise RuntimeError(f"A custom option failed with error message:  {e}")
			value = self.__dict__[_name].value
		elif not name in ['make_category_tree', 'categories','categories_srtd' ]:
			raise RuntimeError(f"'{name}' is not a valid options attribute.")

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
	options['accuracy']					= options_item(0, 	"Accuracy of the optimization algorithm. 0 = fast and inaccurate, 3=slow and maximum accuracy", int, 
																'Accuracy', "%s>0",category='Regression')

	options['add_intercept']					= options_item(True,	"If True, adds intercept if not all ready in the data",
																	bool,'Add intercept', [True,False],['Add intercept','Do not add intercept'],category='Regression')
	
	options['arguments']						= options_item(None, 	"A dict or string defining a dictionary in python syntax containing the initial arguments." 
																	"An example can be obtained by printing ll.args.args_d"
																																																																				, [str,dict, list, np.ndarray], 'Initial arguments')	

	options['ARMA_constraint']		        = options_item(1000,'Maximum absolute value of ARMA coefficients', float, 'ARMA coefficient constraint',
																	 '%s>0', None,category='ARIMA-GARCH')	

	options['constraints_engine']		        = options_item(True,'Determines whether to use the constraints engine', bool, 'Uses constraints engine',
																		[True,False],['Use constraints','Do not use constraints'],category='Regression')	


	options['multicoll_threshold_report']	 = options_item(30,	'Threshold for reporting multicoll problems', float, 'Multicollinearity threshold',
																	 '%s>0',None)		

	options['multicoll_threshold_max']	    = options_item(1000,'Threshold for imposing constraints on collineary variables', float, 'Multicollinearity threshold',
																	'%s>0',None)			

	options['EGARCH']		            = options_item(False,'Normal GARCH, as opposed to EGARCH if True', bool, 'Estimate GARCH directly',
																[True,False],['Direct GARCH','Usual GARCH'],category='ARIMA-GARCH')	



	options['fixed_random_group_eff']			= options_item(0,	'No, fixed or random group effects', int, 'Group fixed random effect',[0,1,2], 
																		['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	options['fixed_random_time_eff']			= options_item(0,	'No, fixed or random time effects', int, 'Time fixed random effect',[0,1,2], 
																		['No effects','Fixed effects','Random effects'],category='Fixed-random effects')
	options['fixed_random_variance_eff']		= options_item(0,	'No, fixed or random group effects for variance', int, 'Variance fixed random effects',[0,1,2], 
																		['No effects','Fixed effects','Random effects'],category='Fixed-random effects')



	options['custom_model']						= options_item(None,	"Custom model class. Must be a class with porperties and methods as definedin the documentation. "
																, type,"Custom model class", category='Regression')
	
	options['include_initvar']					= options_item(True,	"If True, includes an initaial variance term",
																	 	bool,'Include initial variance', [True,False],['Include','Do not include'],category='Regression')

	options['initial_arima_garch_params']	 = options_item(0.1,	'The initial size of arima-garch parameters (all directions will be attempted', 
																	float, 'initial size of arima-garch parameters',
																																																																									 "%s>=0",category='ARIMA-GARCH')		

	options['kurtosis_adj']					= options_item(0,	'Amount of kurtosis adjustment', float, 'Amount of kurtosis adjustment',
																"%s>=0",category='ARIMA-GARCH')	

	options['GARCH_assist']					= options_item(0,	'Amount of weight put on assisting GARCH variance to be close to squared residuals', float, 'GARCH assist',
																"%s>=0",category='ARIMA-GARCH')		

	options['min_group_df']					= options_item(1, "The smallest permissible number of observations in each group. Must be at least 1", int, 
																'Minimum degrees of freedom', "%s>0",category='Regression')

	options['max_iterations']				= options_item(150, "Maximum number of iterations", int, 'Maximum number of iterations', "%s>0",category='Regression')
	

	options['pqdkm']							= options_item([1,1,0,1,1], 
															"ARIMA-GARCH parameters:",int, 'ARIMA-GARCH orders',
																"%s>=0",
																descr_for_input_boxes=["Auto Regression order (ARIMA, p)",
																												"Moving Average order (ARIMA, q)",
																"difference order (ARIMA, d)",
																"Variance Moving Average order (GARCH, k)",
																"Variance Auto Regression order (GARCH, m)"],category='Regression')

	options['robustcov_lags_statistics']		= options_item([100,30],	"Numer of lags used in calculation of the robust \ncovariance matrix for the time dimension", 
																			int, 'Robust covariance lags (time)', "%s>1", 
																			descr_for_input_boxes=["# lags in final statistics calulation","# lags iterations (smaller saves time)"],
																			category='Output')

	options['subtract_means']					= options_item(False,	"If True, subtracts the mean of all variables. This may be a remedy for multicollinearity"
											  							" if the mean is not of interest.",
																		bool,'Subtract means', [True,False],['Subtracts the means','Do not subtract the means'],
																		category='Regression')

	options['supress_output']					= options_item(True,	"If True, no output is printed.",
																		bool,'Supress output', [True,False],
																		['Supresses output','Do not supress output'],category='Regression')

	options['tobit_limits']					= options_item([None,None],	"Determines the limits in a tobit regression. Element 0 is lower limit and element1 is upper limit. "
																		"If None, the limit is not active", 
																		[float,type(None)], 'Tobit-model limits', ['%s>0',None], 
																		descr_for_input_boxes=['lower limit','upper limit'])

	options['tolerance']						= options_item(0.0001, 	"Tolerance. When the maximum absolute value of the gradient divided by the hessian diagonal"
																		"is smaller than the tolerance, the procedure is "
																		"Tolerance in maximum likelihood",
																		float,"Tolerance", "%s>0")	
	
	options['ARMA_round']						= options_item(14, 	"Number og digits to round elements in the ARMA matrices by. Small differences in these values can "
																	"change the optimization path and makes the estimate less robust"
																	"Number of significant digits in ARMA",
																	int,"# of signficant digits", "%s>0")	  

	options['variance_RE_norm']				= options_item(0.000005, 	"This parameter determines at which point the log function "
											   							"involved in the variance RE/FE calculations, "
																		"will be extrapolate by a linear function for smaller values",
																		float,"Variance RE/FE normalization point in log function", "%s>0")		

	options['user_constraints']				= options_item(None,	"Constraints on the regression coefficient estimates. Must be a dict with groups of coefficients "
											   						"where each element can either be None (no constraint), a tuple with a range (min, max) or a single lenght list "
																	"as a float representing a fixed constraint. Se example in README.md. You can extract the arguments dict from "
																	" `result.args`, and substitute the elements with range restrictions or None, or remove groups." 
																	"If you for example put in the dict in `result.args` as it is, you will restrict all parameters "
																	"to be equal to the result.",
																	[str,dict], 'User constraints')

	options['use_analytical']			= options_item(1,	'Use analytical Hessian', int, 'Analytical Hessian',[0,1,2], 
															['No analytical','Analytical in some iterations','Analytical in all iterations'],
															category='Genereal')



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


