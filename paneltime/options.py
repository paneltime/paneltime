#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional

import numpy as np


def is_bool(value):
	return isinstance(value, (bool, np.bool_))


def is_positive(value):
	return value > 0


def is_nonnegative(value):
	return value >= 0


def is_positive_int(value):
	return isinstance(value, int) and value > 0


def is_nonnegative_int(value):
	return isinstance(value, int) and value >= 0


def tuple_of_ints(length, minimum=0):
	def validate(value):
		return (
			isinstance(value, tuple)
			and len(value) == length
			and all(isinstance(item, int) and item >= minimum for item in value)
		)
	return validate


def one_of(values):
	return lambda value: value in values


def positive_or_none_pair(value):
	return (
		isinstance(value, tuple)
		and len(value) == 2
		and all(item is None or item > 0 for item in value)
	)


def is_constraints(value):
	return value is None or isinstance(value, (str, dict))


def is_class_or_none(value):
	return value is None or isinstance(value, type)


@dataclass(frozen=True)
class OptionSpec:
	"""Single source of truth for a public option and its engine mapping."""

	default: Any
	description: str
	domain: Any
	category: str
	engine_name: Optional[str] = None
	example: Any = None
	default_display: Any = None
	validator: Optional[Callable[[Any], bool]] = None


OPTION_SPECS = {
	'fit.order': OptionSpec((1, 1, 0), 'ARIMA order `(p, d, q)`.', 'Three non-negative integers.', 'ARIMA-GARCH', 'pqdkm', '(2, 1, 2)', validator=tuple_of_ints(3)),
	'fit.garch_order': OptionSpec((1, 1), 'GARCH order `(k, m)`.', 'Two non-negative integers.', 'ARIMA-GARCH', 'pqdkm', '(2, 2)', validator=tuple_of_ints(2)),
	'fit.vol': OptionSpec('GARCH', 'Volatility model.', ['GARCH', 'EGARCH'], 'ARIMA-GARCH', 'EGARCH', 'EGARCH', validator=one_of(['GARCH', 'EGARCH'])),
	'fit.effects': OptionSpec(None, 'Fixed or random effects in the mean and variance models.', 'An Effects instance.', 'Effects', default_display='Effects()'),
	'fit.optimizer': OptionSpec(None, 'Numerical optimizer settings.', 'An OptimizerOptions instance.', 'Optimizer', default_display='OptimizerOptions()'),
	'fit.constraints': OptionSpec(None, 'Restrictions on coefficient estimates.', 'None, a string, or a dictionary.', 'Regression', 'user_constraints', validator=is_constraints),
	'fit.add_intercept': OptionSpec(True, 'Add an intercept when one is not present in the formula.', 'True or False.', 'Regression', 'add_intercept', validator=is_bool),
	'fit.subtract_means': OptionSpec(False, 'Subtract variable means before estimation.', 'True or False.', 'Regression', 'subtract_means', validator=is_bool),
	'fit.include_initvar': OptionSpec(False, 'Include an initial variance term.', 'True or False.', 'Regression', 'include_initvar', validator=is_bool),
	'fit.tobit_limits': OptionSpec((None, None), 'Lower and upper limits for a Tobit model.', 'Two positive numbers or None.', 'Regression', 'tobit_limits', '(0, None)', validator=positive_or_none_pair),
	'fit.suppress_output': OptionSpec(True, 'Suppress optimizer progress output.', 'True or False.', 'Output', 'supress_output', validator=is_bool),
	'fit.likelihood': OptionSpec(None, 'Custom likelihood model class.', 'None or a likelihood class.', 'Regression', 'custom_model', validator=is_class_or_none),
	'fit.h_function': OptionSpec(None, 'Custom heteroskedasticity function.', 'None or a supported h-function.', 'Regression'),
	'effects.group': OptionSpec('none', 'Group effect specification.', ['none', 'fixed', 'random'], 'Effects', 'fixed_random_group_eff', validator=one_of(['none', 'fixed', 'random'])),
	'effects.time': OptionSpec('none', 'Time effect specification.', ['none', 'fixed', 'random'], 'Effects', 'fixed_random_time_eff', validator=one_of(['none', 'fixed', 'random'])),
	'effects.variance': OptionSpec('none', 'Variance effect specification.', ['none', 'fixed', 'random'], 'Effects', 'fixed_random_variance_eff', validator=one_of(['none', 'fixed', 'random'])),
	'optimizer.tolerance': OptionSpec(0.0001, 'Tolerance used by the numerical optimizer.', 'Positive number.', 'Optimizer', 'tolerance', '1e-5', validator=is_positive),
	'optimizer.max_iterations': OptionSpec(150, 'Maximum number of optimization iterations.', 'Positive integer.', 'Optimizer', 'max_iterations', '300', validator=is_positive_int),
	'optimizer.accuracy': OptionSpec(0, 'Optimization accuracy level.', 'Non-negative integer.', 'Optimizer', 'accuracy', '1', validator=is_nonnegative_int),
	'optimizer.initial_arima_garch_params': OptionSpec(0.1, 'Initial size of ARIMA-GARCH parameters.', 'Non-negative number.', 'Optimizer', 'initial_arima_garch_params', validator=is_nonnegative),
	'optimizer.arma_constraint': OptionSpec(3, 'Maximum absolute value of ARMA coefficients.', 'Positive number.', 'ARIMA-GARCH', 'ARMA_constraint', validator=is_positive),
	'optimizer.arma_round': OptionSpec(14, 'Number of significant digits used in ARMA matrices.', 'Positive integer.', 'ARIMA-GARCH', 'ARMA_round', validator=is_positive_int),
	'optimizer.garch_min': OptionSpec(1e-12, 'Minimum absolute value of GARCH coefficients.', 'Positive number.', 'ARIMA-GARCH', 'GARCH_min', validator=is_positive),
	'optimizer.garch_assist': OptionSpec(0, 'Weight assigned to the assisting GARCH variance.', 'Non-negative number.', 'ARIMA-GARCH', 'GARCH_assist', validator=is_nonnegative),
	'optimizer.multicoll_threshold_report': OptionSpec(30, 'Threshold for reporting multicollinearity.', 'Positive number.', 'Optimizer', 'multicoll_threshold_report', validator=is_positive),
	'optimizer.min_group_df': OptionSpec(1, 'Minimum observations allowed in each group.', 'Positive integer.', 'Optimizer', 'min_group_df', validator=is_positive_int),
	'optimizer.robust_cov_lags': OptionSpec((100, 30), 'Lags used for robust covariance calculations.', 'Two integers greater than one.', 'Output', 'robustcov_lags_statistics', '(100, 30)', validator=tuple_of_ints(2, minimum=2)),
	'optimizer.variance_re_norm': OptionSpec(0.000005, 'Normalization point for variance random-effects calculations.', 'Positive number.', 'ARIMA-GARCH', 'variance_RE_norm', validator=is_positive),
	'optimizer.kurtosis_adj': OptionSpec(0, 'Kurtosis adjustment.', 'Non-negative number.', 'ARIMA-GARCH', 'kurtosis_adj', validator=is_nonnegative),
}


def _spec(name):
	return OPTION_SPECS[name]


def _validate_spec(name, value):
	spec = _spec(name)
	if spec.validator is None:
		return
	try:
		valid = spec.validator(value)
	except Exception as error:
		raise ValueError(f'Could not validate option {name}: {value!r}') from error
	if not valid:
		raise ValueError(f'Invalid value for {name}: {value!r}; expected {spec.domain}')


def _engine_defaults():
	defaults = {'arguments': None}
	for spec in OPTION_SPECS.values():
		if spec.engine_name is None or spec.engine_name in defaults:
			continue
		defaults[spec.engine_name] = spec.default
	defaults['pqdkm'] = list(_spec('fit.order').default + _spec('fit.garch_order').default)
	defaults['EGARCH'] = _spec('fit.vol').default.upper() == 'EGARCH'
	for name in ('group', 'time', 'variance'):
		effect_spec = _spec(f'effects.{name}')
		defaults[effect_spec.engine_name] = {'none': 0, 'fixed': 1, 'random': 2}[effect_spec.default]
	defaults['robustcov_lags_statistics'] = list(_spec('optimizer.robust_cov_lags').default)
	defaults['tobit_limits'] = list(_spec('fit.tobit_limits').default)
	return defaults


def _option(name):
	spec = _spec(name)
	metadata = {
		'description': spec.description,
		'domain': spec.domain,
		'category': spec.category,
		'engine_name': spec.engine_name,
		'example': spec.example,
	}
	return field(default=spec.default, metadata=metadata)


def _option_factory(name, factory):
	spec = _spec(name)
	metadata = {
		'description': spec.description,
		'domain': spec.domain,
		'category': spec.category,
		'engine_name': spec.engine_name,
		'example': spec.example,
	}
	return field(default_factory=factory, metadata=metadata)


class EngineOptions:
	"""Mutable settings consumed by the numerical engine."""

	def __init__(self):
		object.__setattr__(self, '_allowed_names', {
			spec.engine_name for spec in OPTION_SPECS.values() if spec.engine_name is not None
		} | {'arguments'})
		for name, value in _engine_defaults().items():
			object.__setattr__(self, name, value)

	def __setattr__(self, name, value):
		if name not in self._allowed_names:
			raise AttributeError(f"'{name}' is not a valid engine option")
		object.__setattr__(self, name, value)

	def validate(self):
		if len(self.pqdkm) != 5 or any(not isinstance(value, int) or value < 0 for value in self.pqdkm):
			raise ValueError('pqdkm must contain five non-negative integers')
		if any(value not in (0, 1, 2) for value in (
				self.fixed_random_group_eff, self.fixed_random_time_eff, self.fixed_random_variance_eff)):
			raise ValueError('fixed/random effect settings must be 0, 1, or 2')
		if self.accuracy < 0:
			raise ValueError('accuracy must be non-negative')
		if self.max_iterations <= 0 or self.min_group_df <= 0:
			raise ValueError('max_iterations and min_group_df must be positive')
		if any(value <= 0 for value in (
				self.ARMA_constraint, self.GARCH_min, self.multicoll_threshold_report,
				self.tolerance, self.variance_RE_norm, self.ARMA_round)):
			raise ValueError('engine constraint, tolerance, and normalization settings must be positive')
		if any(value < 0 for value in (
				self.initial_arima_garch_params, self.kurtosis_adj, self.GARCH_assist)):
			raise ValueError('engine parameter magnitudes must be non-negative')
		if len(self.robustcov_lags_statistics) != 2 or any(
				not isinstance(value, int) or value <= 1 for value in self.robustcov_lags_statistics):
			raise ValueError('robustcov_lags_statistics must contain two integers greater than one')
		if len(self.tobit_limits) != 2 or any(
				value is not None and value <= 0 for value in self.tobit_limits):
			raise ValueError('tobit_limits must contain two positive values or None')
		if self.user_constraints is not None and not isinstance(self.user_constraints, (str, dict)):
			raise TypeError('user_constraints must be a string, dict, or None')
		if self.custom_model is not None and not isinstance(self.custom_model, type):
			raise TypeError('custom_model must be a class or None')
		if self.arguments is not None and not isinstance(self.arguments, (str, dict, list, np.ndarray)):
			raise TypeError('arguments must be a string, dict, list, array, or None')
		for name in ('add_intercept', 'EGARCH', 'include_initvar', 'subtract_means', 'supress_output'):
			if not isinstance(getattr(self, name), (bool, np.bool_)):
				raise TypeError(f'{name} must be boolean')
		return self


def create_engine_options():
	return EngineOptions()


@dataclass
class Effects:
	"""Fixed or random effects included in the mean or variance model."""

	group: str = _option('effects.group')
	time: str = _option('effects.time')
	variance: str = _option('effects.variance')

	def validate(self):
		for option in fields(Effects):
			_validate_spec(f'effects.{option.name}', getattr(self, option.name))


@dataclass
class OptimizerOptions:
	"""Numerical optimizer settings used by :meth:`Model.fit`."""

	tolerance: float = _option('optimizer.tolerance')
	max_iterations: int = _option('optimizer.max_iterations')
	accuracy: int = _option('optimizer.accuracy')
	initial_arima_garch_params: float = _option('optimizer.initial_arima_garch_params')
	arma_constraint: float = _option('optimizer.arma_constraint')
	arma_round: int = _option('optimizer.arma_round')
	garch_min: float = _option('optimizer.garch_min')
	garch_assist: float = _option('optimizer.garch_assist')
	multicoll_threshold_report: float = _option('optimizer.multicoll_threshold_report')
	min_group_df: int = _option('optimizer.min_group_df')
	robust_cov_lags: tuple[int, int] = _option('optimizer.robust_cov_lags')
	variance_re_norm: float = _option('optimizer.variance_re_norm')
	kurtosis_adj: float = _option('optimizer.kurtosis_adj')

	def validate(self):
		for option in fields(OptimizerOptions):
			_validate_spec(f'optimizer.{option.name}', getattr(self, option.name))


@dataclass
class FitOptions:
	"""Per-call public configuration for a paneltime model fit."""

	order: tuple[int, int, int] = _option('fit.order')
	garch_order: tuple[int, int] = _option('fit.garch_order')
	vol: str = _option('fit.vol')
	effects: Effects = _option_factory('fit.effects', Effects)
	optimizer: OptimizerOptions = _option_factory('fit.optimizer', OptimizerOptions)
	constraints: Any = _option('fit.constraints')
	add_intercept: bool = _option('fit.add_intercept')
	subtract_means: bool = _option('fit.subtract_means')
	include_initvar: bool = _option('fit.include_initvar')
	tobit_limits: tuple[Optional[float], Optional[float]] = _option('fit.tobit_limits')
	suppress_output: bool = _option('fit.suppress_output')
	likelihood: Any = _option('fit.likelihood')
	h_function: Any = _option('fit.h_function')

	def validate(self):
		for name in ('order', 'garch_order', 'vol', 'constraints', 'add_intercept',
				'subtract_means', 'include_initvar', 'tobit_limits', 'suppress_output', 'likelihood'):
			_validate_spec(f'fit.{name}', getattr(self, name))
		self.effects.validate()
		self.optimizer.validate()

	def to_engine_options(self):
		self.validate()
		engine_options = create_engine_options()
		engine_options.pqdkm = list(self.order) + list(self.garch_order)
		engine_options.EGARCH = self.vol.upper() == 'EGARCH'
		effect_values = {'none': 0, 'fixed': 1, 'random': 2}
		for name in ('group', 'time', 'variance'):
			spec = _spec(f'effects.{name}')
			setattr(engine_options, spec.engine_name, effect_values[getattr(self.effects, name)])
		for name in ('constraints', 'add_intercept', 'subtract_means', 'include_initvar', 'tobit_limits', 'suppress_output', 'likelihood'):
			spec = _spec(f'fit.{name}')
			value = getattr(self, name)
			if name == 'tobit_limits':
				value = list(value)
			setattr(engine_options, spec.engine_name, value)
		for option in fields(OptimizerOptions):
			spec = _spec(f'optimizer.{option.name}')
			setattr(engine_options, spec.engine_name, getattr(self.optimizer, option.name))
		return engine_options.validate()


def option_schema():
	"""Return documentation metadata for the public fit options."""
	for key, spec in OPTION_SPECS.items():
		group, name = key.split('.', 1)
		yield {
			'group': group,
			'name': name,
			'default': spec.default if spec.default_display is None else spec.default_display,
			'description': spec.description,
			'domain': spec.domain,
			'category': spec.category,
			'example': spec.example or '',
			'validator': None if spec.validator is None else spec.validator.__name__,
		}