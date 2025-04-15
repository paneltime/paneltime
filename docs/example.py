import wbdata
import paneltime as pt

# Define variables to download
indicators = {
    'NY.GDP.MKTP.KD.ZG': 'GDP_growth',    # BNP-vekst
    'FP.CPI.TOTL.ZG': 'Inflation',        # Inflasjon (konsumprisindeks)
    'FR.INR.LEND': 'Interest_rate',        # Rentenivå (lånerente)
	'NY.GNS.ICTR.ZS': 'Gross_Savings',  # Gross savings (% of GDP)
    'NE.CON.GOVT.ZS': 'Gov_Consumption',  # Government consumption (% of GDP)
}


# Download data
df = wbdata.get_dataframe(indicators)

#avoiding extreme interest rates
df = df[abs(df['Inflation'])<30]

# Run the regression:
pt.options.pqdkm = (1, 1, 1, 1, 1)
m = pt.execute('Inflation~L(Gross_Savings)+L(Inflation)+L(Interest_rate)+D(L(Gov_Consumption))'
					 , df, timevar = 'date',idvar='country' )

# display results
print(m)