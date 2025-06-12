import paneltime as pt
import loadwb


# loading data
df = loadwb.load_worldbank_data()

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# define ARIMA-GARCH lag structure and model:
pt.options.pqdkm = (2, 2, 1, 2, 2)
pt.options.EGARCH = True

#Run the regression:
m = pt.execute('Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))'
					 , df, timevar = 'date',idvar='country' )

# display results
print(m)
a=0