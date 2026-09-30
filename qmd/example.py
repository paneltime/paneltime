import paneltime as pt
import loadwb


# loading data
df = loadwb.load_worldbank_data()

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# Fit an ARIMA(2, 1, 2) EGARCH(2, 2) model:
model = pt.Model(
	'Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))',
	df,
	time='date',
	entity='country',
)
m = model.fit(order=(2, 1, 2), garch_order=(2, 2), vol='EGARCH')

# display results
print(m)