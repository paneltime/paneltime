import numpy as np
from abc import ABC, abstractmethod


def _h_identity(e, e2, v):
	"""Picklable replacement for `lambda e, e2, v: e2`"""
	return e2


def _h_log(e, e2, v):
	"""Picklable replacement for `lambda e, e2, v: np.log(e2)`"""
	return np.log(e2)


class LikelihoodModel(ABC):
	"""Base class for user-defined likelihood models.

	Subclasses may implement only :meth:`loglike`; numerical score and Hessian
	methods are then supplied by finite differences.
	"""

	def __init__(self, e, init_var, a=0, k=0, z=None):
		self.e, self.z = np.asarray(e), z
		self.a, self.k = a, k
		self.variance_bounds(init_var)
		self.variance_definitions()
		self.set_h_function()
		missing = [name for name in ('var', 'e', 'z', 'v', 'v_inv', 'var_pos')
				   if not hasattr(self, name)]
		if missing:
			raise ValueError('LikelihoodModel is missing required attributes: ' + ', '.join(missing))

	@abstractmethod
	def variance_bounds(self, init_var):
		"""Set ``var`` and its valid-region mask ``var_pos``."""

	@abstractmethod
	def variance_definitions(self):
		"""Set the variance representation used by the likelihood."""

	def set_h_function(self):
		"""Set the heteroskedasticity function and its derivatives."""
		self.h = _h_identity
		self.h_val = self.h(self.e, self.e ** 2 + 1e-8, self.v)
		self.h_val_cpp = ''
		self.h_e_val = 2 * self.e
		self.h_2e_val = np.full_like(self.e, 2.0)
		self.h_z_val = self.h_2z_val = self.h_ez_val = None

	@abstractmethod
	def loglike(self):
		"""Return pointwise log-likelihood values."""

	def score(self):
		"""Return a finite-difference score when no analytic score is supplied."""
		return _finite_difference(self, 'loglike', first=True)

	def hessian(self):
		"""Return a finite-difference Hessian when no analytic Hessian is supplied."""
		return _finite_difference(self, 'loglike', first=False)

	def ll(self):
		return self.loglike()

	def dll(self):
		return self.score()

	def ddll(self):
		return self.hessian()


def _finite_difference(model, method, first):
	"""Numerically differentiate a pointwise likelihood with respect to e/var."""
	h = 1e-6
	original_e = np.array(model.e, copy=True)
	original_var = np.array(model.var, copy=True)
	def evaluate(e, var):
		model.e = e
		model.var = var
		model.variance_definitions()
		return np.asarray(getattr(model, method)())
	try:
		base = evaluate(original_e, original_var)
		e_plus = evaluate(original_e + h, original_var)
		e_minus = evaluate(original_e - h, original_var)
		if first:
			return (e_plus - e_minus) / (2 * h)
		return (e_plus - 2 * base + e_minus) / (h * h)
	finally:
		model.e = original_e
		model.var = original_var




LL_CONST =-0.5*np.log(2*np.pi)
	
class Exponential:
	def __init__(self, e, init_var, a, k, z):

		# Do not change any variable names here, as they are used elsewhere
		self.k, self.a, self.z, self.e = k, a, z, e

		self.variance_bounds(init_var)
		self.variance_definitions()
		self.set_h_function()

	def variance_bounds(self, init_var):
		# Allways define a range for the variance. For an expnential model, minvar may be negative.
		self.minvar = -500
		self.maxvar = 500

		# self.var is the variance measure exposed to the optimization algorithm.
		# you can define other measures in internal calculations. In this expample self.v and self.v_inv are used internally.
		# Make sure this is reflected in the derivatives with respect to the exposed variance measure.

		self.var = np.maximum(np.minimum(init_var, self.maxvar), self.minvar)

		# var_pos defines when the variance boundaries are active, in which case the derivatives are zero.
		# You need to muptiply the variance derivatives with this variable in the dll and ddll functions to get 
		# correct derivatives.
		self.var_pos=(init_var < self.maxvar) * (init_var > self.minvar)

	def variance_definitions(self):
		(var, e, z) = (self.var, self.e, self.z)

		# variance and its inverse:
		self.v = np.exp(var)
		self.v_inv = np.exp(-var)
		
		# variance innovation	:
		self.e2 = e**2 + 1e-8

	def set_h_function(self):
		"Defines the heteroskedasticity function and its derivatives"
		(e, e2, v, z) = (self.e, self.e2, self.v, self.z)

		# === Main function definition ===
		# Define the h function definition:
		self.h = _h_log

		# Define the h funciton value (do not change):
		self.h_val = self.h(e, e2, v)

		# TNo need to insert the c++ equivalent of self.h_val as it is hard coded.
		self.h_val_cpp = ''

		# === Derivatives of the h function ===
		self.h_e_val = 2*e/(e2)
		self.h_2e_val = (2/(e2) - 4*e**2/(e2**2))
		self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None


	def ll(self):
		"Calculates the log-likelihood value for the model."
		(e, e2, v, var, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.var, self.v_inv, self.a, self.k, self.var_pos, self.z)

		LL_CONST =-0.5*np.log(2*np.pi)
		
		ll = LL_CONST-0.5*(var + e2*v_inv)
		
		return ll

	def dll(self):
		"Calculates the first derivatives of the log-likelihood function."
		(e, e2, v, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.a, self.k, self.var_pos, self.z)

		dll_e = -(e*v_inv)
		dll_var = -0.5*(1-(e2*v_inv))

		dll_var *= var_pos

		return dll_var, dll_e
	
	def ddll(self):
		"Calculates the second derivatives of the log-likelihood function."
		(e, e2, v, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.a, self.k, self.var_pos, self.z)

		d2ll_de2 = -v_inv
		d2ll_dvar_de = e*v_inv
		d2ll_dvar2 = -0.5*((e2*v_inv))

		d2ll_dvar_de*=var_pos
		d2ll_dvar2*=var_pos

		return d2ll_de2, d2ll_dvar_de, d2ll_dvar2


EGARCH = Exponential


class Hyperbolic:
	def __init__(self, e, v, a, k, z):

		# Do not change any variable names here, as they are used elsewhere
		self.a = a
		self.k = k
		self.z = z

		# Allways define a range for the variance. For an expnential model, minvar may be negative.
		self.minvar = 1e-30
		self.maxvar = 1e+30

		self.var = np.maximum(np.minimum(v, self.maxvar), self.minvar)

		# var_pos defines when the variance boundaries are active, in which case the derivatives are zero.
		# You need to muptiply the variance derivatives with this variable in the dll and ddll functions to get 
		# correct derivatives.
		self.var_pos=(v < self.maxvar) * (v > self.minvar)

		# Defining verious variables:
		self.v = self.var
		self.v_inv = 1/v
		
		self.e = e
		self.e2 = e**2 + 1e-8

		# Defining the heteroskedasticity function:
		self.set_h_function()

	def set_h_function(self):
		"Defines the heteroskedasticity function and its derivatives"
		(e, e2, v, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.a, self.k, self.var_pos, self.z)
	
		# === Main function definition ===
		# Define the h function definition:
		self.h = _h_identity

		
		# Define the h funciton value (do not change):
		self.h_val = self.h(e, e2, v)
		
		# The insert the c++ equivalent of self.h_val below.
		self.h_val_cpp = ''#Hyperbolic is handled explicitly in c++ code

		self.h_e_val = 2*e
		self.h_2e_val = 2
		self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None

	def ll(self):

		(e, e2, v, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.a, self.k, self.var_pos, self.z)

		ll = LL_CONST-0.5*(np.log(v)+(1-k)*e2/v
			+ a* (np.abs(e2-v)*v)
			+ (k/3)* e2**2*v**2
			)
		
		return ll

	def dll(self):

		(e, e2, v, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.a, self.k, self.var_pos, self.z)


		dll_e   =-0.5*(	(1-k)*2*e/v	)

		dll_e   +=-0.5*(a* 2*np.sign(e2-v)*e/v
						+ (k/3)* 4*e2*e/v**2
						)
		
		dll_var =-0.5*(	1/v-(1-k)*e2/v**2	)
		
		dll_var +=-0.5*(- a* (np.sign(e2-v)/v)
						- a* (np.abs(e2-v)/v**2)
						- (k/3)* 2*e2**2/v**3
								)
		dll_var *= var_pos

		return dll_var, dll_e
		



	def ddll(self):
		
		(e, e2, v, v_inv, a, k, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.a, self.k, self.var_pos, self.z)


		d2ll_de2 	 =-0.5*(	(1-k)*2/v	)
		d2ll_de2 	 +=-0.5*(a* 2*np.sign(e2-v)/v
							+ (k/3)* 12*e2/v**2
								)
		
		d2ll_dv_de =-0.5*(	-(1-k)*2*e/v**2)
		d2ll_dv_de +=-0.5*(- a* 2*np.sign(e2-v)*e/v**2
							- (k/3)* 8*e2*e/v**3
								)
		
		d2ll_dv2 	 =-0.5*(-1/v**2+(1-k)*2*e2/v**3	)
		d2ll_dv2 	 +=-0.5*(a* (np.sign(e2-v)/v**2)
							+ a* 2*(np.abs(e2-v)/v**3)
							+ a* (np.sign(e2-v)/v**2)
							+ (k/3)* 6*e2**2/v**4
								)
		
		d2ll_dv_de *= var_pos
		d2ll_dv2 *= var_pos

		return d2ll_de2, d2ll_dv_de, d2ll_dv2
	



class Normal:
	def __init__(self, e, v, a, k, z):

		# Do not change any variable names here, as they are used elsewhere
		self.z = z

		# Allways define a range for the variance. For an expnential model, minvar may be negative.
		self.minvar = 1e-30
		self.maxvar = 1e+30

		self.var = np.maximum(np.minimum(v, self.maxvar), self.minvar)

		# var_pos defines when the variance boundaries are active, in which case the derivatives are zero.
		# You need to muptiply the variance derivatives with this variable in the dll and ddll functions to get 
		# correct derivatives.
		self.var_pos=(v < self.maxvar) * (v > self.minvar)

		# Defining verious variables:
		self.v = self.var
		self.v_inv = 1/self.var
		
		self.e = e
		self.e2 = e**2 + 1e-8

		# Defining the heteroskedasticity function:
		self.set_h_function()

	def set_h_function(self):
		"Defines the heteroskedasticity function and its derivatives"
		(e, e2, v, v_inv, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.var_pos, self.z)
	
		# === Main function definition ===
		# Define the h function definition:
		self.h = _h_identity

		
		# Define the h funciton value (do not change):
		self.h_val = self.h(e, e2, v)
		
		# The insert the c++ equivalent of self.h_val below.
		self.h_val_cpp = ''#Hyperbolic is handled explicitly in c++ code

		self.h_e_val = 2*e
		self.h_2e_val = 2
		self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None

	def ll(self):

		(e, e2, v, v_inv, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.var_pos, self.z)

		ll = LL_CONST-0.5*(np.log(v)+e2/v)
		
		return ll

	def dll(self):

		(e, e2, v, v_inv, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.var_pos, self.z)


		dll_e   = - e/v
		
		dll_var = -0.5*(1/v - e2/v**2)
		
		dll_var *= var_pos

		return dll_var, dll_e
		



	def ddll(self):
		
		(e, e2, v, v_inv, var_pos, z) = (
			self.e, self.e2, self.v, self.v_inv, self.var_pos, self.z)


		d2ll_de2 	 =-0.5*(	2/v	)
		
		d2ll_dv_de =-0.5*(	-2*e/v**2)

		d2ll_dv2 	 =-0.5*(-1/v**2+2*e2/v**3	)

		d2ll_dv_de *= var_pos
		d2ll_dv2 *= var_pos

		return d2ll_de2, d2ll_dv_de, d2ll_dv2
	


