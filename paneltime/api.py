"""The public model and results API."""

from dataclasses import replace
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd

from . import main
from .options import Effects, FitOptions, OptimizerOptions


class RandomEffectsResult:
	"""Estimated panel random effects and their residual standard deviations."""

	def __init__(self, legacy):
		self.group = getattr(legacy, 'residuals_i', None)
		self.time = getattr(legacy, 'residuals_t', None)
		self.group_std = getattr(legacy, 'residuals_std_i', None)
		self.time_std = getattr(legacy, 'residuals_std_t', None)


class Results:
	"""Results from a fitted :class:`Model` model.

	The legacy summary is retained privately to keep formatting and diagnostic
	output compatible while the commonly used statistics are exposed directly.
	"""

	def __init__(self, legacy):
		self._legacy = legacy
		self._output = legacy.output
		self._table = legacy.table
		self.optim_result = legacy.general
		self._names = list(legacy.names.captions)
		# Transitional aliases for code written against Summary v1.
		self.names = legacy.names
		self.results = legacy.results
		self.general = legacy.general
		self.ll = legacy.ll
		data = legacy.results
		self._params = self._series(data.params)
		self._bse = self._series(data.se)
		self._tvalues = self._series(data.tstat)
		self._pvalues = self._series(data.tsign)
		self.random_effects = RandomEffectsResult(legacy.random_effects)

	def _series(self, value):
		if value is None:
			return pd.Series(index=self._names, dtype=float)
		return pd.Series(np.asarray(value), index=self._names, dtype=float)

	@property
	def params(self) -> pd.Series:
		"""Estimated coefficients indexed by variable name."""
		return self._params

	@property
	def bse(self) -> pd.Series:
		"""Robust coefficient standard errors indexed by variable name."""
		return self._bse

	@property
	def tvalues(self) -> pd.Series:
		"""Coefficient t statistics indexed by variable name."""
		return self._tvalues

	@property
	def pvalues(self) -> pd.Series:
		"""Two-sided coefficient p values indexed by variable name."""
		return self._pvalues

	@property
	def nobs(self) -> int:
		"""Number of observations used in estimation."""
		return int(self._legacy.panel.NT)

	@property
	def df_resid(self) -> int:
		"""Residual degrees of freedom."""
		return int(self._legacy.panel.df)

	@property
	def llf(self) -> float:
		"""Maximized log-likelihood value."""
		return float(getattr(self._legacy.general, 'log_likelihood', self._legacy.general.comm.f))

	@property
	def aic(self) -> float:
		"""Akaike information criterion."""
		return -2 * self.llf + 2 * len(self.params)

	@property
	def bic(self) -> float:
		"""Bayesian information criterion."""
		return -2 * self.llf + np.log(self.nobs) * len(self.params)

	@property
	def converged(self) -> bool:
		"""Whether the optimizer reported convergence."""
		return bool(self._legacy.general.converged)

	@property
	def resid(self):
		"""Residuals from the fitted model."""
		return self._legacy.results.residuals

	@property
	def fittedvalues(self):
		"""Fitted values corresponding to the model sample."""
		return np.asarray(self._legacy.panel.Y).reshape(-1) - np.asarray(self.resid).reshape(-1)

	def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
		"""Return coefficient confidence intervals.

		Parameters
		----------
		alpha : float
			Significance level between zero and one.
		"""
		if not 0 < alpha < 1:
			raise ValueError('alpha must be between 0 and 1')
		low = np.asarray(self._table.d.get('conf_low', np.full(len(self.params), np.nan)))
		high = np.asarray(self._table.d.get('conf_high', np.full(len(self.params), np.nan)))
		return pd.DataFrame({0: low, 1: high}, index=self.params.index)

	def summary(self):
		"""Return the legacy formatted summary object."""
		return self._legacy

	def predict(self, signals=None):
		"""Predict observations, optionally using heteroskedasticity signals."""
		return self._legacy.predict(signals)

	def forecast(self, steps: int = 1):
		"""Forecast future observations for the requested number of steps."""
		if not isinstance(steps, int) or steps < 1:
			raise ValueError('steps must be a positive integer')
		return np.asarray(self.predict())[-steps:]

    
class Model:
	"""Panel ARIMA/GARCH model using a constructor followed by ``fit``."""

	def __init__(self, formula: str, data: pd.DataFrame, entity: Optional[str] = None,
			 time: Optional[str] = None):
		if not isinstance(data, pd.DataFrame) or data.empty:
			raise ValueError('data must be a non-empty pandas DataFrame')
		self.formula = formula
		self.data = data
		self.entity = entity
		self.time = time

	def fit(self, order=(1, 1, 0), garch_order=(1, 1), vol='GARCH',
			effects=None, cov_type='robust', optimizer=None, constraints=None,
			likelihood=None, h_function=None, _legacy_options=None,
			_het_factors=None, _instruments=None, add_intercept=True,
			subtract_means=False, include_initvar=False,
			tobit_limits=(None, None), suppress_output=True) -> Results:
		"""Fit the model and return a :class:`Results` instance."""
		if _legacy_options is not None:
			legacy = _legacy_options
			entity, time = self.entity, self.time
			result = main.execute(self.formula, self.data, time, entity,
				_het_factors, legacy, None, None, _instruments, True, None)
			return Results(result)
		effects = effects or Effects()
		optimizer = optimizer or OptimizerOptions()
		config = FitOptions(order=order, garch_order=garch_order, vol=vol,
			effects=effects, optimizer=optimizer, constraints=constraints,
			likelihood=likelihood, h_function=h_function,
			add_intercept=add_intercept, subtract_means=subtract_means,
			include_initvar=include_initvar, tobit_limits=tobit_limits,
			suppress_output=suppress_output)
		legacy = config.to_legacy()
		window = None
		exe_tab = None
		if isinstance(self.data.index, pd.MultiIndex):
			entity = self.entity or self.data.index.names[0]
			time = self.time or self.data.index.names[1]
		else:
			entity, time = self.entity, self.time
		if likelihood is not None:
			legacy.custom_model = likelihood
		result = main.execute(self.formula, self.data, time, entity, None,
			legacy, window, exe_tab, None, True, None)
		return Results(result)
