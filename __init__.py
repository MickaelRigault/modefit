
__version__ = "0.1.0"



try:
    from astrobject.astrobject.baseobject import BaseObject
except ImportError:
    raise ImportError("modefit module is based upon astrobject. install it: https://github.com/MickaelRigault/astrobject.git")

     
from fitter import *
