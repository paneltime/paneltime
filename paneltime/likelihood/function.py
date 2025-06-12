#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from . import models

HFUNC_ITEMS = ['h_val', 'h_e_val', 'h_2e_val', 'h_z_val', 'h_2z_val', 'h_ez_val']
EXPORT_ITEMS = ['var', 'e', 'e2']

class LLFunction:
	def __init__(self, panel, e, v, z, hyp = False):

		model = self.get(panel, hyp)

		a = panel.options.GARCH_assist
		k = panel.options.kurtosis_adj

		incl = panel.included[3]



		with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

			self.model = model(e, v, a, k, z)

		for key, x in [('a', a), ('k', k), ('incl', incl)]:
			setattr(self, key, x)
			setattr(self.model, key, x)

		_incl = incl==0
		for key in self.model.__dict__:
			x = self.model.__dict__[key]
			if key in EXPORT_ITEMS+HFUNC_ITEMS:
				if type(x) == np.ndarray:
					x[_incl] = 0
				elif not x is None:
					x = x*incl
				setattr(self.model, key, x)
				setattr(self, key, x)


		self.v_inv05 = self.model.v**0.5

		self.test()

		self.ll_value = self.ll()





	def get(self, panel, hyp):
		"Selects model based on options. Called by the panel module."

		if not panel.options.custom_model is None:
			model = panel.options.custom_model
		elif panel.options.EGARCH == 0:
			a = panel.options.GARCH_assist
			k = panel.options.kurtosis_adj
			if a == 0 and k == 0 and not hyp:
				model = models.Normal
			else:
				model = models.Hyperbolic
		elif panel.options.EGARCH==1:
			model = models.Exponential

		return model

	def ll(self):

		with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
			self.ll_value = self.model.ll()

			self.ll_value[self.incl==0] = 0

			return self.ll_value




	def gradient(self):


		with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

			dLL_var, DLL_e = self.model.dll()

			dLL_var[self.incl==0], DLL_e[self.incl==0] = 0, 0

			return dLL_var, DLL_e



	def hessian(self):

		with np.errstate(divide='ignore', invalid='ignore'):

			d2LL_de2, d2LL_dln_de, d2LL_dln2 = self.model.ddll()

			d2LL_de2[self.incl==0], d2LL_dln_de[self.incl==0], d2LL_dln2[self.incl==0] = 0, 0, 0

			return d2LL_de2, d2LL_dln_de, d2LL_dln2
		


	def test(self):

		model = self.model
		for k in HFUNC_ITEMS:
			if k not in model.__dict__:
				raise ValueError(
					f"The model must contain the following properties: {', '.join(HFUNC_ITEMS)}."
					"If the shape parameter is not used, its derivatives 'h_z', 'h_z2', and 'h_e_z' may be set to "
					"None. If you are using a custom model, please ensure that the model class has these properties."
				)


	def z_active(self):
		"""Check if the z-derivative keys are present in the h_func dictionary."""
		if not np.all((getattr(self.model,k, None) == None) 
					or (getattr(self.model,k, '') == '') 
						for k in HFUNC_ITEMS[4:]  ):
			return True
		return False