"""The public model and results API."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from importlib import import_module

from . import main
from .options import Effects, FitOptions, OptimizerOptions
from .random_effects import REObj


MP = None



class RandomEffectsResult:
	"""Estimated panel random effects and their residual standard deviations."""

	def __init__(self, effects):
		self.group = getattr(effects, 'residuals_i', None)
		self.time = getattr(effects, 'residuals_t', None)
		self.group_std = getattr(effects, 'residuals_std_i', None)
		self.time_std = getattr(effects, 'residuals_std_t', None)


class EffectEstimate:
	"""Estimated effects and variance component for one panel dimension."""

	def __init__(self, mode, estimates, variance=None):
		self.mode = mode
		self.estimates = estimates
		self.variance = variance
		self.std = None if variance is None else float(np.sqrt(max(variance, 0)))


class PanelEffectResults:
	"""Conventional centered group and time effects from the fitted residuals."""

	def __init__(self, panel, likelihood):
		self.group, self.time = self._components(panel, likelihood)

	@staticmethod
	def _components(panel, likelihood):
		group_mode_value = panel.options.fixed_random_group_eff
		time_mode_value = panel.options.fixed_random_time_eff
		group_mode = {0: 'none', 1: 'fixed', 2: 'random'}[group_mode_value]
		time_mode = {0: 'none', 1: 'fixed', 2: 'random'}[time_mode_value]

		if panel.pqdkm[2] > 0:
			return EffectEstimate(group_mode, None), EffectEstimate(time_mode, None)

		residuals = np.asarray(likelihood.u)[..., 0]
		mask = np.asarray(panel.included[3])[..., 0].astype(bool)
		unit_count, period_count = residuals.shape

		group_values = np.zeros(unit_count)
		time_values = np.zeros(period_count)

		group_variance = None
		time_variance = None
		group_weights = np.ones(unit_count)
		time_weights = np.ones(period_count)

		if group_mode == 'random':
			re_obj_group = REObj(panel, True, panel.T_i, panel.T_i, 2)
			re_obj_group.RE(likelihood.u, panel)
			group_variance = float(max(re_obj_group.v_var, 0))
			group_e_var = float(np.asarray(re_obj_group.e_var).reshape(-1)[0])
			t_i = np.asarray(panel.T_i).reshape(-1)
			group_weights = group_variance / (group_variance + group_e_var / t_i)

		if time_mode == 'random':
			re_obj_time = REObj(panel, False, panel.date_count_mtrx, panel.date_count, 2)
			re_obj_time.RE(likelihood.u, panel)
			time_variance = float(max(re_obj_time.v_var, 0))
			time_e_var = float(np.asarray(re_obj_time.e_var).reshape(-1)[0])
			date_count = np.asarray(panel.date_count).reshape(-1)
			time_weights = time_variance / (time_variance + time_e_var / date_count)

		for _ in range(1000):
			previous_group = group_values.copy()
			previous_time = time_values.copy()

			if group_mode != 'none':
				for unit in range(unit_count):
					valid = mask[unit]
					if np.any(valid):
						mean_residual = np.mean(residuals[unit, valid] - time_values[valid])
						if group_mode == 'random':
							group_values[unit] = group_weights[unit] * mean_residual
						else:
							group_values[unit] = mean_residual
				group_values -= np.mean(group_values)

			if time_mode != 'none':
				for period in range(period_count):
					valid = mask[:, period]
					if np.any(valid):
						mean_residual = np.mean(residuals[valid, period] - group_values[valid])
						if time_mode == 'random':
							time_values[period] = time_weights[period] * mean_residual
						else:
							time_values[period] = mean_residual
				time_values -= np.mean(time_values)

			if max(
				float(np.max(np.abs(group_values - previous_group))),
				float(np.max(np.abs(time_values - previous_time))),
			) < 1e-10:
				break

		group_estimates = None
		time_estimates = None
		if group_mode != 'none':
			group_labels = np.asarray(panel.original_names)
			group_estimates = pd.Series(group_values, index=group_labels, name='group_effect')
		if time_mode != 'none':
			time_frame = panel.input.timevar
			time_labels = pd.unique(time_frame.iloc[:, 0])
			time_estimates = pd.Series(time_values, index=time_labels, name='time_effect')

		group_result = EffectEstimate(group_mode, group_estimates, group_variance)
		time_result = EffectEstimate(time_mode, time_estimates, time_variance)
		return group_result, time_result


class Summary:
	"""Summary from a fitted :class:`Model` model.

	The engine summary components remain available as categorized attributes,
	while common statistics are exposed directly on this object.
	"""

	def __init__(self, summary):
		self._summary = summary
		self.output = summary.output
		self.table = summary.table
		self.count = summary.count
		self.optim_result = summary.general
		self._names = list(summary.names.captions)
		# Compatibility aliases for consumers of the engine summary fields.
		self.names = summary.names
		self.results = summary.results
		self.general = summary.general
		self.ll = summary.ll
		data = summary.results
		self._params = self._series(data.params)
		self._bse = self._series(data.se)
		self._tvalues = self._series(data.tstat)
		self._pvalues = self._series(data.tsign)
		self.random_effects = RandomEffectsResult(summary.random_effects)
		self.names_grouped = summary.panel.args.names_d
		self.names_dependent = summary.output.stats.info.dep_var
		self.names_independents = summary.panel.args.names_d['beta']
		self.pqdkm = summary.panel.pqdkm
		self.stats = summary.output.stats
		self.sign_codes = summary.panel.sign_codes
		self.options = summary.panel.options
		self.panel = summary.panel
		self.prediction_names = summary.prediction_names
		self.effect_results = PanelEffectResults(summary.panel, summary.ll)

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
		"""Conventional coefficient standard errors indexed by variable name."""
		return self._bse

	@property
	def robust_bse(self) -> pd.Series:
		"""Robust sandwich standard errors indexed by variable name."""
		values = self.results.coef_se_robust
		if values is None:
			return pd.Series(index=self._names, dtype=float)
		return pd.Series(np.asarray(values), index=self._names, dtype=float)

	def bse_at(self, params: dict | pd.Series, robust: bool = False) -> pd.Series:
		"""Evaluate standard errors at supplied values without optimizing."""
		if not isinstance(robust, (bool, np.bool_)):
			raise TypeError('robust must be a bool')
		diagnostics = self.bse_at_diagnostics(params)
		return diagnostics['robust_bse' if robust else 'bse']

	def bse_at_diagnostics(self, params: dict | pd.Series) -> dict:
		"""Return both SE variants and diagnostics evaluated at supplied values.

		The score is reported as total and per included observation. Hessian
		conditioning is computed on the free parameters after diagonal scaling.
	"""
		if not isinstance(params, (dict, pd.Series)):
			raise TypeError('params must be a dict or pandas Series keyed by parameter name')

		names = list(self.params.index)
		missing = [name for name in names if name not in params]
		extra = [name for name in params.keys() if name not in names]
		if missing or extra:
			raise ValueError(f'Parameter names must match the fitted model; missing={missing}, extra={extra}')
		values = np.asarray([params[name] for name in names], dtype=float)
		if not np.all(np.isfinite(values)):
			raise ValueError('All parameter values must be finite')

		from .likelihood.main import LL
		from .maximization.computation import Computation
		from .output.output import grand_mean_variance, sandwich

		panel = self.panel
		comput = Computation(values, panel, panel.options.tolerance, 4 * np.finfo(float).eps, False)
		ll = LL(values, panel, constraints=comput.constr)
		if ll.LL is None:
			raise ValueError('Likelihood is undefined at the supplied parameter values')

		gradient, scores = comput.calc_gradient(ll)
		hessian = comput.calc_hessian(ll)
		if hessian is None or not np.all(np.isfinite(hessian)):
			raise ValueError('Hessian is undefined at the supplied parameter values')

		lags = panel.options.robustcov_lags_statistics[1]
		se_robust, se_unadjusted, _, _ = sandwich(
			hessian, scores, gradient, comput.constr, panel, lags
		)
		se_robust_opposite, se_unadjusted_opposite, _, _ = sandwich(
			hessian, scores, gradient, comput.constr, panel, lags, oposite=True
		)
		se_robust = np.asarray(se_robust, dtype=float)
		se_unadjusted = np.asarray(se_unadjusted, dtype=float)
		se_robust_opposite = np.asarray(se_robust_opposite, dtype=float)
		se_unadjusted_opposite = np.asarray(se_unadjusted_opposite, dtype=float)
		se_robust[np.isnan(se_robust)] = se_robust_opposite[np.isnan(se_robust)]
		se_unadjusted[np.isnan(se_unadjusted)] = se_unadjusted_opposite[np.isnan(se_unadjusted)]

		extra_variance = grand_mean_variance(panel, ll)
		if panel.input.has_intercept and extra_variance > 0:
			intercept_index = panel.args.positions['beta'][0]
			se_robust[intercept_index] = np.sqrt(se_robust[intercept_index] ** 2 + extra_variance)
			se_unadjusted[intercept_index] = np.sqrt(se_unadjusted[intercept_index] ** 2 + extra_variance)

		free = np.ones(len(names), dtype=bool)
		free[list(comput.constr.fixed)] = False
		free_indices = np.flatnonzero(free)
		hessian_free = np.asarray(hessian)[np.ix_(free, free)]
		if hessian_free.size:
			scale = np.sqrt(np.maximum(np.abs(np.diag(hessian_free)), 1e-300))
			scaled = -hessian_free / np.outer(scale, scale)
			scaled = 0.5 * (scaled + scaled.T)
			eigenvalues, eigenvectors = np.linalg.eigh(scaled)
			absolute_eigenvalues = np.abs(eigenvalues)
			largest = float(absolute_eigenvalues.max())
			condition_index = (
				float(np.sqrt(largest / max(float(absolute_eigenvalues.min()), largest * 1e-30)))
				if largest > 0 else float('nan')
			)
			weakest = eigenvectors[:, int(np.argmin(absolute_eigenvalues))]
			weak_direction = {names[index]: float(value) for index, value in zip(free_indices, weakest)}
			hessian_condition = float(np.linalg.cond(hessian_free))
		else:
			eigenvalues = np.array([], dtype=float)
			condition_index = hessian_condition = float('nan')
			weak_direction = {}

		included = np.asarray(panel.included[3], dtype=bool)
		var_pos = np.asarray(ll.llfunc.model.var_pos, dtype=bool)
		included_count = int(included.sum())
		variance_clipped_fraction = (
			float(np.count_nonzero((~var_pos) & included) / included_count)
			if included_count else float('nan')
		)
		max_observation_score = np.max(np.abs(scores), axis=(0, 1))
		return {
			'log_likelihood': float(ll.LL),
			'bse': pd.Series(se_unadjusted, index=names, dtype=float),
			'robust_bse': pd.Series(se_robust, index=names, dtype=float),
			'score': pd.Series(gradient, index=names, dtype=float),
			'mean_score': pd.Series(gradient / panel.NT, index=names, dtype=float),
			'max_observation_score': pd.Series(max_observation_score, index=names, dtype=float),
			'hessian_diagonal': pd.Series(np.diag(hessian), index=names, dtype=float),
			'hessian_condition_number': hessian_condition,
			'scaled_condition_index': condition_index,
			'scaled_hessian_eigenvalues': absolute_eigenvalues.tolist() if hessian_free.size else [],
			'weakest_direction': weak_direction,
			'variance_clipped_fraction': variance_clipped_fraction,
		}

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
		return int(self._summary.panel.NT)

	@property
	def df_resid(self) -> int:
		"""Residual degrees of freedom."""
		return int(self._summary.panel.df)

	@property
	def llf(self) -> float:
		"""Maximized log-likelihood value."""
		return float(getattr(self._summary.general, 'log_likelihood', self._summary.general.comm.f))

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
		return bool(self._summary.general.converged)

	@property
	def resid(self):
		"""Residuals from the fitted model."""
		return self._summary.results.residuals

	@property
	def fittedvalues(self):
		"""Fitted values corresponding to the model sample."""
		return np.asarray(self._summary.panel.Y).reshape(-1) - np.asarray(self.resid).reshape(-1)

	def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
		"""Return coefficient confidence intervals.

		Parameters
		----------
		alpha : float
			Significance level between zero and one.
		"""
		if not 0 < alpha < 1:
			raise ValueError('alpha must be between 0 and 1')
		from .output.stat_dist import tinv
		critical_value = tinv(1 - alpha / 2, self.df_resid)
		margin = critical_value * self.robust_bse
		return pd.DataFrame({0: self.params - margin, 1: self.params + margin}, index=self.params.index)

	def summary(self):
		"""Return this categorized summary object."""
		return self

	def __str__(self):
		"""Return the formatted regression summary."""
		return str(self._summary)

	def latex(self):
		"""Return the regression table formatted as LaTeX."""
		return self._summary.latex()

	def html(self):
		"""Return the regression table formatted as HTML."""
		return self._summary.html()

	def results_table(self, fmt='CONSOLE'):
		"""Return the formatted coefficient table."""
		return self._summary.results_table(fmt)

	def statistics(self):
		"""Return the model statistics section."""
		return self._summary.statistics()

	def diagnostics(self):
		"""Return the diagnostics section."""
		return self._summary.diagnostics()

	def accounting(self):
		"""Return the degrees-of-freedom accounting section."""
		return self._summary.accounting()

	def predict(self, signals=None):
		"""Predict observations, optionally using heteroskedasticity signals."""
		return self._summary.predict(signals)

	def forecast(self, steps: int = 1):
		"""Return future dependent-variable forecasts for each panel unit."""
		if not isinstance(steps, (int, np.integer)) or isinstance(steps, (bool, np.bool_)) or steps < 1:
			raise ValueError('steps must be a positive integer')
		steps = int(steps)
		prediction_name = f'Predicted {self.panel.input.Y_names[0]}'
		forecasts = self.predict()[prediction_name].dropna()
		if forecasts.empty:
			raise ValueError('No future predictions are available')
		by_unit = forecasts.groupby(level=0, sort=False)
		available_steps = int(by_unit.size().min())
		if steps > available_steps:
			word = 'step is' if available_steps == 1 else 'steps are'
			raise ValueError(f'Only {available_steps} future {word} available for at least one panel unit')
		return by_unit.head(steps).to_numpy()


    
class Model:
	"""Panel ARIMA/GARCH model using a constructor followed by ``fit``."""

	def __init__(self, formula: str, data: pd.DataFrame, entity: Optional[str] = None,
			 time: Optional[str] = None, multiprocess = False):
		global MP
		if not isinstance(data, pd.DataFrame) or data.empty:
			raise ValueError('data must be a non-empty pandas DataFrame')
		self.formula = formula
		self.data = data
		self.entity = entity
		self.time = time
		if multiprocess and MP is None:
			try:
				paneltime_mp = import_module('paneltime_mp')
			except ImportError as error:
				raise ImportError('multiprocess=True requires the optional paneltime_mp package') from error
			MP = paneltime_mp.Master(7)
		self.mp = MP

	def fit(self, order=(1, 1, 0), garch_order=(1, 1), vol='GARCH',
			effects=None, cov_type='robust', optimizer=None, constraints=None,
			likelihood=None, h_function=None, add_intercept=True,
			subtract_means=False, include_initvar=False,
			tobit_limits=(None, None), suppress_output=True) -> Summary:
		"""Fit the model and return a :class:`Summary` instance."""
		effects = effects or Effects()
		optimizer = optimizer or OptimizerOptions()
		config = FitOptions(order=order, garch_order=garch_order, vol=vol,
			effects=effects, optimizer=optimizer, constraints=constraints,
			likelihood=likelihood, h_function=h_function,
			add_intercept=add_intercept, subtract_means=subtract_means,
			include_initvar=include_initvar, tobit_limits=tobit_limits,
			suppress_output=suppress_output)
		engine_options = config.to_engine_options()
		window = None
		exe_tab = None
		if isinstance(self.data.index, pd.MultiIndex):
			entity = self.entity or self.data.index.names[0]
			time = self.time or self.data.index.names[1]
		else:
			entity, time = self.entity, self.time
		summary = main.execute(self.formula, self.data, time, entity, None,
			engine_options, window, exe_tab, None, True, self.mp)
		return Summary(summary)
