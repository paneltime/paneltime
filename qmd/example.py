
import paneltime as pt
import os
import pandas as pd
import time
import loadwb
import numpy as np



# loading data
df = loadwb.load_worldbank_data()

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# Run the regression:
pt.options.pqdkm = (2, 2, 1, 2, 2)
pt.options.EGARCH = False

class MyModel: #debug
	def __init__(self, e, e2, v, a, k, incl):

		self.args = (e, e2, v, a, k, incl)

		self.v = np.exp(v)
		self.v_inv = np.exp(-v)

		self.b_str = 'x0 = (e)^2+1e-8;    log(x0)'

		self.h_dict = {#exprtk compatible expressions for cpp:
			'h_val': 		'x0 = (e)^2+1e-8;    log(x0)',
			'h_e_val': 		'x0 = (e)^2+1e-8;    e*2/x0',
			'h_e2_val': 	'x0 = (e)^2+1e-8;    2/x0-4*e**2/x0**2',
		}
		
		self.h_add = 1e-8

		self.e2

		self.h_dict = {	'h':    	np.log(e2),
						'h_e_val':  2*e/(e2),
						'h_2e_val':	(2/(e2) - 4*e**2/(e2**2))
			}
		
		self.minvar = 1e-30
		self.maxvar = 1e+30


	def ll(self):
		(e, e2, v, a, k, incl) = self.args
		LL_CONST =-0.5*np.log(2*np.pi)
		ll = LL_CONST-0.5*(v+e2*self.v_inv)
		
		return ll

	def dll(self):
		(e, e2, v, a, k, incl) = self.args

		dll_e=-(e*self.v_inv)
		dll_var=-0.5*(incl-(e2*self.v_inv))	
		
		return dll_var, dll_e
		
	def ddll(self):
		(e, e2, v, a, k, incl) = self.args

		d2ll_de2=-self.v_inv
		d2ll_dln_de=e*self.v_inv
		d2ll_dln2=-0.5*e2*self.v_inv

		return d2ll_de2, d2ll_dln_de, d2ll_dln2

if False:
	
	pt.options.custom_model = MyModel

if False:#
	pt.options.h_dict ={
		'h':      'x0 = e^2 + 1e-8; x0^z',
		'h_e':    'x0 = e^2 + 1e-8; z * x0^(z - 1) * 2 * e',
		'h_e2':   'x0 = e^2 + 1e-8; z * (4 * e^2 * x0^(z - 2) * (z - 1) + 2 * x0^(z - 1))',
		'h_z':    'x0 = e^2 + 1e-8; x0^z * log(x0)',
		'h_z2':   'x0 = e^2 + 1e-8; x0^z * (log(x0))^2',
		'h_e_z':  'x0 = e^2 + 1e-8; 2 * e * (z * x0^(z - 1) * log(x0) + x0^(z - 1))'
	}
# The h function avoids division by zero in the GARCH model using e2 + (e2==0)*1e-18.
# The math of the h function is handled by `exprtk`. 
# Refer to https://paneltime.github.io/paneltime/html/hfunc.html for `exprtk` syntax details.


#pt.options.EGARCH = True
t0 = time.time()
m = pt.execute('Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))'
					 , df, timevar = 'date',idvar='country' )

print(f"Time taken: {time.time()-t0} seconds")

# display results
print(m)
a=0