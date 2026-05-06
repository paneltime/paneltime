
import paneltime as pt
import loadwb
import numpy as np



# loading data
df = loadwb.load_worldbank_data()

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# define ARIMA-GARCH lag structure:
pt.options.pqdkm = (2, 2, 1, 2, 2)
pt.options.EGARCH = False
# Define a custom model class:
class MyModel:
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
		self.h = lambda e, e2, v: np.log(e2)

		# Define the h funciton value (do not change):
		self.h_val = self.h(e, e2, v)

		# The insert the c++ equivalent of self.h_val below.
		self.h_val_cpp = 'log(e2)'

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

# Assign the custom model to the relevant option in paneltime:
pt.options.custom_model = MyModel


m = pt.execute('Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))'
					 , df, timevar = 'date',idvar='country' )


print(m)
