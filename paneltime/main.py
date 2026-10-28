"""Prepare panel data, fit locally, and construct the final results."""

import time
import warnings
from types import SimpleNamespace

import numpy as np

from .maximization import Summary
from .maximization.dfpmax import dfpmax
from .processing import model_parser
from .processing.arma_start import ARMAInitializationWarning
from .processing.panel import Panel


np.set_printoptions(suppress=True, precision=8)


def execute(model_string, dataframe, timevar, idvar, heteroscedasticity_factors,
            options, instruments=None):
    """Run a local fit, preserving numerical warnings and final inference."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warnings.simplefilter('default', ARMAInitializationWarning)
        datainput = Input(
            dataframe.copy(), model_string, idvar, timevar, options,
            heteroscedasticity_factors, instruments,
        )
        if not options.supress_output:
            print("Creating panel")
        panel = Panel(datainput, options)
        panel.init()

        start_time = time.time()
        result = SimpleNamespace(**dfpmax(panel.args.args_init.args_v, panel))
        if not options.supress_output:
            print(f"Iterations: {result.its}, LL: {result.f}; {result.msg}")
        return Summary(result, panel, start_time)


class Input:
    def __init__(self, dataframe, model_string, idvar, timevar, options,
                 heteroscedasticity_factors, instruments):
        model_parser.get_variables(
            self, dataframe, model_string, idvar, timevar,
            heteroscedasticity_factors, instruments, options,
        )
        self.descr = model_string
        self.args = options.arguments if options.arguments != "" else None
