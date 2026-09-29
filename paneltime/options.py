#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

def create_options(deprecated=False):
	options = options_dict()
	opt = OptionsObj(options, deprecated=deprecated)
	return opt

def options_to_txt():

	options = options_dict()
	a = []

	for o in options:
		opt = options[o]
		if isinstance(opt.dtype, list):
			tp = [i.__name__ for i in opt.dtype]
		else:
			tp = opt.dtype.__name__
		value = opt.value
		if isinstance(value, str):
			value = value.replace('\n','<br>').replace('\t','a&#9;')
			if len(value)>12:
				value = value[:9]+"..."
		perm = opt.permissible_values
		if perm == None:
			perm = 'Any'
		a.append([o, value, tp, perm, f"<b>{opt.name}:</b> {opt.description}".replace('\n','<br>').replace('\t','a&#9;')])

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
		if isinstance(dtype, str):
			self.dtype_str=dtype
		elif isinstance(dtype, (list, tuple)):
			self.dtype_str=str(dtype).replace('<class ','').replace('[','').replace(']','').replace('>','').replace("'",'')
		else:
			self.dtype_str= 'NA'

		self.permissible_values=permissible_values
		self.value_description=value_description
		self.descr_for_input_boxes=descr_for_input_boxes
		self.category=category
		self.name=name
		self.selection_var= len(descr_for_input_boxes)==0 and isinstance(permissible_values, list)
		self.is_inputlist=len(self.descr_for_input_boxes)>0



	def set(self,value):
		self.valid(value)
		if str(self.value)!=str(value):
			self.value=value

	def valid(self,value):
		if self.permissible_values is None:
			if self.dtype is type:
				isclass = isinstance(value, self.dtype)
				if not isclass:
					raise TypeError(f"Expected type 'type' (class type) for option {self.code_name} but got {type(value)}")
				return
			try:
				if self.dtype(value)==value:
					return
			except Exception as e:
				raise RuntimeError(f'Checking correct type of {self.code_name} failed with error message: {e}')
			if isinstance(value, tuple(self.dtype)):
				return
			else:
				raise TypeError(f'Cannot set option {self.code_name}, expected type {self.dtype}, got {type(value)} ')
		self.valid_test(value, self.permissible_values)


	def valid_test(self,value,permissible):
		if permissible is None:
			return True
		if isinstance(permissible, (list, tuple)):
			try:
				if not isinstance(value, list):
					value=self.dtype(value)
					if value in permissible:
						return
					else:
						raise RuntimeError(f'Setting option {self.code_name} failed. Value {value} not in permissible values {permissible}')
				else:
					valid=True
					for i in range(len(value)):
						dtype = self.dtype[i] if isinstance(self.dtype, (list, tuple)) else self.dtype
						if value[i] is None:
							continue
						value[i] = dtype(value[i])
						if permissible[i] is not None:
							valid = valid * eval(permissible[i] % value[i])
			except Exception as e:
				raise RuntimeError(f'Setting option {self.code_name} failed with error message: {e}')
			return valid
		elif isinstance(permissible, str):
			if isinstance(value, (list, tuple)):
				return np.all([eval(permissible %(i,)) for i in value])
			else:
				return eval(permissible %(value,))
		else:
			print('No method to handle this permissible')

class OptionsObj:
	def __init__(self, options, deprecated=False):
		super().__setattr__('_deprecated', deprecated)
		super().__setattr__('_warned', False)
		for o in options:
			super().__setattr__('_' + o, options[o]) 
			super().__setattr__(o, options[o].value) 

		self.make_category_tree()

	def __setattr__(self, name, value):
		# Trigger a custom function when an attribute is set
		_name = '_' + name
		if _name in self.__dict__:
			if self.__dict__.get('_deprecated') and not self.__dict__.get('_warned'):
				warnings.warn(
					'pt.options is deprecated; pass configuration to model.fit() instead.',
					DeprecationWarning, stacklevel=2)
				super().__setattr__('_warned', True)
			self.__dict__[_name].set(value)
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
			is_object = i[0]=='_'
			# No options item can have underscore in the beginning of its name, as it defines
			# the internal object version of the option
			if is_object and isinstance(opt[i], options_item):
				if opt[i].category in d:
					d[opt[i].category].append(opt[i])
				else:
					d[opt[i].category]=[opt[i]]
				opt[i].code_name=i[1:]
		self.categories=d	
		keys=np.array(list(d.keys()))
		self.categories_srtd=keys[keys.argsort()]


@dataclass
class Effects:
	"""Fixed or random effects included in the mean or variance model."""

	group: str = 'none'
	time: str = 'none'
	variance: str = 'none'

	def validate(self):
		for name in ('group', 'time', 'variance'):
			value = getattr(self, name)
			if value not in {'none', 'fixed', 'random'}:
				raise ValueError(f"effects.{name} must be 'none', 'fixed', or 'random'; got {value!r}")


@dataclass
class OptimizerOptions:
	"""Numerical optimizer settings used by :meth:`Model.fit`."""

	tolerance: float = 0.0001
	max_iterations: int = 150
	accuracy: int = 0
	initial_arima_garch_params: float = 0.1
	arma_constraint: float = 3
	arma_round: int = 14
	garch_min: float = 1e-12
	garch_assist: float = 0
	multicoll_threshold_report: float = 30
	min_group_df: int = 1
	robust_cov_lags: tuple[int, int] = (100, 30)
	variance_re_norm: float = 0.000005
	kurtosis_adj: float = 0

	def validate(self):
		if self.tolerance <= 0 or self.max_iterations <= 0:
			raise ValueError('optimizer.tolerance and optimizer.max_iterations must be positive')
		if self.accuracy < 0:
			raise ValueError('optimizer.accuracy must be non-negative')
		if any(value < 0 for value in (self.initial_arima_garch_params, self.garch_min, self.garch_assist, self.kurtosis_adj)):
			raise ValueError('optimizer parameter magnitudes must be non-negative')


@dataclass
class FitOptions:
	"""Per-call public configuration for a paneltime model fit."""

	order: tuple[int, int, int] = (1, 1, 0)
	garch_order: tuple[int, int] = (1, 1)
	vol: str = 'GARCH'
	effects: Effects = field(default_factory=Effects)
	optimizer: OptimizerOptions = field(default_factory=OptimizerOptions)
	constraints: Any = None
	add_intercept: bool = True
	subtract_means: bool = False
	include_initvar: bool = False
	tobit_limits: tuple[Optional[float], Optional[float]] = (None, None)
	suppress_output: bool = True
	likelihood: Any = None
	h_function: Any = None

	def validate(self):
		if len(self.order) != 3 or any(not isinstance(value, int) or value < 0 for value in self.order):
			raise ValueError('order must be a tuple of three non-negative integers')
		if len(self.garch_order) != 2 or any(not isinstance(value, int) or value < 0 for value in self.garch_order):
			raise ValueError('garch_order must be a tuple of two non-negative integers')
		if self.vol.upper() not in {'GARCH', 'EGARCH'}:
			raise ValueError("vol must be either 'GARCH' or 'EGARCH'")
		self.effects.validate()
		self.optimizer.validate()

	def to_legacy(self):
		self.validate()
		legacy = create_options()
		legacy.pqdkm = list(self.order) + list(self.garch_order)
		legacy.EGARCH = self.vol.upper() == 'EGARCH'
		legacy.fixed_random_group_eff = {'none': 0, 'fixed': 1, 'random': 2}[self.effects.group]
		legacy.fixed_random_time_eff = {'none': 0, 'fixed': 1, 'random': 2}[self.effects.time]
		legacy.fixed_random_variance_eff = {'none': 0, 'fixed': 1, 'random': 2}[self.effects.variance]
		if self.constraints is not None:
			legacy.user_constraints = self.constraints
		legacy.add_intercept = self.add_intercept
		legacy.subtract_means = self.subtract_means
		legacy.include_initvar = self.include_initvar
		if self.tobit_limits != (None, None):
			legacy.tobit_limits = list(self.tobit_limits)
		legacy.supress_output = self.suppress_output
		optimizer_map = {
			'tolerance': 'tolerance', 'max_iterations': 'max_iterations', 'accuracy': 'accuracy',
			'initial_arima_garch_params': 'initial_arima_garch_params',
			'arma_constraint': 'ARMA_constraint', 'arma_round': 'ARMA_round', 'garch_min': 'GARCH_min',
			'garch_assist': 'GARCH_assist', 
			'multicoll_threshold_report': 'multicoll_threshold_report', 'min_group_df': 'min_group_df',
			'robust_cov_lags': 'robustcov_lags_statistics', 'variance_re_norm': 'variance_RE_norm',
			'kurtosis_adj': 'kurtosis_adj'}
		for source, target in optimizer_map.items():
			setattr(legacy, target, getattr(self.optimizer, source))
		if self.likelihood is not None:
			legacy.custom_model = self.likelihood
		return legacy



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

	options['ARMA_constraint']		        = options_item(3,'Maximum absolute value of ARMA coefficients', float, 'ARMA coefficient constraint',
																	 '%s>0', None,category='ARIMA-GARCH')	
	options['GARCH_min']		        = options_item(1e-12,'Minimum absolute value of GARCH coefficients', float, 'GARCH coefficient constraint',
																	 '%s>0', None,category='ARIMA-GARCH')	


	options['multicoll_threshold_report']	 = options_item(30,	'Threshold for reporting multicoll problems', float, 'Multicollinearity threshold',
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
	
	options['include_initvar']					= options_item(False,	"If True, includes an initaial variance term",
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





	return options

