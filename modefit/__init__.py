""" Classes and Functions allowing advance fitting techniques. """

from unimodal import *
from bimodal  import *
from hubble   import get_hubblefit
from basics import get_polyfit,get_normpolyfit
try:
    from emissionlines import linefit
except:
    warning.warns("Cannot import emissionlines. Most likely you do not have astrobject.")
