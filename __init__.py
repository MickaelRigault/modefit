"""  Perform various kind on fit based on the same structure.

This module is based on Minuit for the fit and has  ambedded tools to
use scipy minimizer similarly.
Parameters can have boundaries and can be fixed.

An Bayesian framework, making use of MCMC algorihtms are automatically
implemented in any functions.
"""


from fitter import *

__version__ = "0.2.1"
