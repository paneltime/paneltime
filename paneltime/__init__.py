#!/usr/bin/env python
# -*- coding: utf-8 -*-



from .output import formatting

from . import info

from .api import Model, Summary
from .options import Effects, FitOptions, OptimizerOptions, OPTION_SPECS, option_schema

def format(summaries, heading, caption, col_headings = [], variable_groups = {}, digits=3, fmt='latex', size = 1, fpath = None):
	"""Prints the results of a set of summaries .\n"""
	s = formatting.format(summaries, fmt, heading, col_headings, variable_groups, digits, size, caption, fpath)

	return s


__version__ = info.version



