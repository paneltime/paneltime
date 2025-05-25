import numpy as np


LL_CONST =-0.5*np.log(2*np.pi)

class Exponential: 
	def __init__(self, e, v, a, k, incl):
		self.args = (e, a, k, incl)

		self.minvar = -500
		self.maxvar = 500

		self.var = incl*np.maximum(np.minimum(v, self.maxvar), self.minvar)
		self.dvar_pos=(v < self.maxvar) * (v > self.minvar)

		self.v = incl*np.exp(self.var)
		self.v_inv = incl*np.exp(-self.var)

		self.e2 = e**2 + 1e-8


		self.h_val = np.log(self.e2)*incl
		self.h_val_cpp = b''#This is the c++ equivalent of self.h_val, but EGARCH is handled explicitly in c++ code

		self.h_e_val = 2*incl*e/(self.e2)
		self.h_2e_val = incl*(2/(self.e2) - 4*e**2/(self.e2**2))
		self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None		


	def ll(self):
		(e, a, k, incl) = self.args

		ll = (LL_CONST-0.5*(self.var+self.e2*self.v_inv))*incl
		
		return ll

	def dll(self):
		(e,  a, k, incl) = self.args

		dll_e = -(e*self.v_inv)*incl
		dll_var = -0.5*(incl-(self.e2*self.v_inv))*incl
		
		dll_var *= self.dvar_pos

		return dll_var, dll_e
		
	def ddll(self):
		(e,  a, k, incl) = self.args

		d2ll_de2 = -self.v_inv*incl
		d2ll_dvar_de = e*self.v_inv*incl
		d2ll_dvar2 = -0.5*((self.e2*self.v_inv))*incl

		d2ll_dvar_de*=self.dvar_pos
		d2ll_dvar2*=self.dvar_pos


		return d2ll_de2, d2ll_dvar_de, d2ll_dvar2
	

class Hyperbolic:
	def __init__(self, e, v, a, k):

		# Do not change any variable names here, as they are used elsewhere

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

		self.vars = (self.e, self.e2, self.v,  self.v_inv, k, a, self.var_pos)

		# Defining the heteroskedasticity function:
		self.set_h_function()

	def set_h_function(self):
		"Defines the heteroskedasticity function and its derivatives"
		(e, e2, v, v_inv, a, k, var_pos) = self.vars
	
		self.h_val = e2

		# Insert the c++ equivalent of self.h_val below. For custom functions this 
		# string is parsed by c++ in the calulation of the ARIMA and GARCH matrices
		# You use 
		self.h_val_cpp = b''#This is the c++ equivalent of self.h_val, but EGARCH is handled explicitly in c++ code

		self.h_e_val = 2*e
		self.h_2e_val = 2
		self.h_z_val, self.h_2z_val,  self.h_ez_val = None,None,None

	def ll(self):

		(e, e2, v, v_inv, a, k, var_pos) = self.vars

		ll = LL_CONST-0.5*(np.log(v)+(1-k)*e2/v
			+ a* (np.abs(e2-v)*v)
			+ (k/3)* e2**2*v**2
			)
		
		return ll

	def dll(self):

		(e, e2, v, v_inv, a, k, var_pos) = self.vars


		Dll_e   =-0.5*(	(1-k)*2*e/v	)

		Dll_e   +=-0.5*(a* 2*np.sign(e2-v)*e/v
						+ (k/3)* 4*e2*e/v**2
						)
		
		dll_var =-0.5*(	1/v-(1-k)*e2/v**2	)
		
		dll_var +=-0.5*(- a* (np.sign(e2-v)/v)
						- a* (np.abs(e2-v)/v**2)
						- (k/3)* 2*e2**2/v**3
								)
		dll_var *= var_pos

		return dll_var, Dll_e
		



	def ddll(self):
		
		(e, e2, v, v_inv, a, k, var_pos) = self.vars


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
	



