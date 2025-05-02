import os
import pandas as pd

TMPFILE = f"{os.path.dirname(__file__)}/wb.dmp"


def load_worldbank_data():
	"""Load World Bank data from a local file or download it if not available."""

	if os.path.exists(TMPFILE):
		return pd.read_pickle(TMPFILE)
	else:
		import wbdata
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

		# Catching
		df.to_pickle(TMPFILE)

		return df

	
