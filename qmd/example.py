
import paneltime as pt
import os
import pandas as pd
import time
import loadwb



# loading data
df = loadwb.load_worldbank_data()

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# Run the regression:
pt.options.pqdkm = (2, 2, 1, 2, 2)

# Defing a custom h function for the GARCH model (identical to standard GARCH model):
pt.options.h_dict = {'h':	'x0 = e^2 + 1e-8 +adfadf(e); '
							'x0 + (x0==0)*1e-18;',
					 'h_e':'2*e', 'h_e2':'2', 'h_z':'','h_z2':'', 'h_e_z':''}
# The h function avoids division by zero in the GARCH model using e2 + (e2==0)*1e-18.
# The math of the h function is handled by `exprtk`. 
# Refer to https://paneltime.github.io/paneltime/docs/hfunc.html for `exprtk` syntax details.


#pt.options.EGARCH = True
t0 = time.time()
m = pt.execute('Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))'
					 , df, timevar = 'date',idvar='country' )

print(f"Time taken: {time.time()-t0} seconds")

# display results
print(m)
a=0