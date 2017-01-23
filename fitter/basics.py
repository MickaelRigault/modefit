#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Basic Filters """

import numpy as np
from scipy.special import orthogonal
from scipy import stats
from .baseobjects import BaseModel, BaseFitter, DataHandler

__all__ = ["get_polyfit", "get_normpolyfit"]


def get_polyfit(x, y, dy, degree,
                legendre=False, **kwargs):
    """ Get the object to fit a `degree`th order polynomes.

    Parameters
    ----------
    x, y, dy: [arrays]
       data to be fitted. Remark that the errors on the fit parameters
       are accurate only if the chi2/dof ~ 1

    degree: [int]
        The degree of the polynome used for model:
        - 1 means horizontal line (b)
        - 2 means slope (ax + b if not legendre)
        - 3 etc.

    legendre: [bool] -optional-
        Shall the model be based on Legendre polynomial or
        simple polynomal (ax+bx**2 + cx**3 etc.)

    Returns
    -------
    PolynomeFit
    """
    return PolynomeFit(np.asarray(x), np.asarray(y),
                       np.asarray(dy), degree,
                       legendre=legendre,
                        **kwargs)

def get_normpolyfit(x, y, dy, degree, ngauss,
                legendre=False, **kwargs):
    """ Get the object to fit a `degree`th order polynomes 
    with `ngauss` gaussian on top of it.

    Parameters
    ----------
    x, y, dy: [arrays]
       data to be fitted. Remark that the errors on the fit parameters
       are accurate only if the chi2/dof ~ 1

    ngauss: [int] 
        Number of gaussian to add. Each has 3 degrees of freedom (mean, sigma, amplitude)

    degree: [int]
        The degree of the polynome used for model:
        - 1 means horizontal line (b)
        - 2 means slope (ax + b if not legendre)
        - 3 etc.

    legendre: [bool] -optional-
        Shall the model be based on Legendre polynomial or
        simple polynomal (ax+bx**2 + cx**3 etc.)

    Returns
    -------
    PolynomeFit
    """
    return NormPolynomeFit(np.asarray(x), np.asarray(y),
                       np.asarray(dy), degree, ngauss,
                       legendre=legendre,
                        **kwargs)

####################################
#                                  #
#                                  #
#    Polynomial Fits               #
#                                  #
#                                  #
####################################
class PolynomeFit( BaseFitter, DataHandler ):
    """ """
    PROPERTIES         = ["xdata"]
    DERIVED_PROPERTIES = ["xscaled"]

    def __init__(self, x, y, dy, degree,
                 names=None, legendre=True):
        """ """
        self.set_data(x, y, dy)
        self.set_model(polynomial_model(degree), use_legendre=legendre)
        
    # ============== #
    #  Main Methods  #
    # ============== #
    # - To Be Defined
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        return self._scaled_xdata if self.model.use_legendre else self.xdata, self.data, self.errors, False
    
    # - Super It
    def set_data(self, x, y, dy=None, names=None):
        """ Basic method to set the data """
        self._properties["xdata"]  = x
        self._derived_properties["xscaled"]  = (x-np.min(x))/(np.max(x)-np.min(x))*2-1.
        super(PolynomeFit, self).set_data(y, errors=dy, names=names)

    def set_model(self, model, use_legendre=False, **kwargs):
        """ """
        super(PolynomeFit, self).set_model(model, **kwargs)
        self.model.use_legendre=use_legendre


    def show(self, savefile=None, show=True, ax=None,
             show_model=True, xrange=None,
             mcmc=False, nsample=100, mlw=2, ecolor="0.3",
             mcmccolor=None, modelcolor= "k", modellw=2, 
             **kwargs):
        """ """
        import matplotlib.pyplot as mpl 
        from astrobject.utils.mpladdon import figout, errorscatter
        from astrobject.utils.tools    import kwargs_update
        self._plot = {}
        if ax is None:
            fig = mpl.figure(figsize=[8,5])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        elif "plot" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. ")
        else:
            fig = ax.figure
        
        # - Basics
        prop = kwargs_update( dict(ms=15, mfc=mpl.cm.Blues(0.6,0.9), mec=mpl.cm.Blues(0.8,0.9),
                                   ls="None",mew=1.5, marker="o", zorder=5), **kwargs)
        
        pl = ax.plot(self.xdata,self.data, **prop)
        er = ax.errorscatter(self.xdata,self.data, dy=self.errors, zorder=prop["zorder"]-1,
                             ecolor=ecolor)
        # - Model
        if show_model:
            if xrange is None:
                xx = np.linspace(self.xdata.min()-np.abs(self.xdata.max()), self.xdata.max()*2, 1000)
            else:
                xx = np.linspace(xrange[0],xrange[1], 1000)
            
            if not mcmc:
                model = ax.plot(xx,self.model.get_model(xx), ls="-", lw=modellw,
                                color=modelcolor, scalex=False, scaley=False, zorder=np.max([prop["zorder"]-2,1]))
                
            elif not self.has_mcmc():
                warnings.warn("No MCMC loaded. use run_mcmc()")
                model = []
            else:
                if mcmccolor is None:
                    mcmccolor = mpl.cm.binary(0.6,0.3)
                model = [ax.plot(xx,self.model.get_model(xx, param=param), color=mcmccolor,
                                scalex=False, scaley=False, zorder=np.max([prop["zorder"]-3,1]))
                        for param in self.mcmc.samples[np.random.randint(len(self.mcmc.samples), size=nsample)]]
                
                model.append(ax.plot(xx,self.model.get_model(xx, param=np.asarray(self.mcmc.derived_values).T[0]), 
                                        ls="-", lw=modellw, color=modelcolor,
                                        scalex=False, scaley=False, zorder=np.max([prop["zorder"]-2,1])))
                
        else:
            model = []
        # ---------
        # - Limits
        # ------
        self._plot['ax']     = ax
        self._plot['figure'] = fig
        self._plot['plot']   = [pl,model]
        self._plot['prop']   = prop
        
        fig.figout(savefile=savefile, show=show)
        return self._plot
    # ================= #
    #  Properties       #
    # ================= #
    @property
    def xdata(self):
        return self._properties["xdata"]
    
    @property
    def _scaled_xdata(self):
        """ You that to fit the data"""
        return self._derived_properties["xscaled"]
    

class NormPolynomeFit( PolynomeFit ):
    """ """
    def __init__(self, x, y, dy, degree, ngauss,
                 names=None, legendre=True):
        """ """
        self.set_data(x, y, dy)
        self.set_model(normal_and_polynomial_model(degree, ngauss),
                           use_legendre=legendre)
        
############################
#                          #
# Model Legendre Polynome  #
#                          #
############################

def polynomial_model(degree):
    """ 
    This function builds and returns the Model used to fit the
    Hubble Data with the defined standardization parameter.
    
    Returns
    -------
    Child of ModelStandardization (with set  STANDARDIZATION)
    """
    class N_PolyModel( PolyModel ):
        DEGREE = degree

    return N_PolyModel()


class PolyModel( BaseModel ):
    """ Virtual Class able to handle any polynomial order fitter """
    DEGREE = 0
    
    PROPERTIES         = ["parameters", "legendre"]
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []

    # ================ #
    #  Main Method     #
    # ================ #
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        
        cls.FREEPARAMETERS = ["a%d"%(i) for i in range(cls.DEGREE)]
        return super(PolyModel,cls).__new__(cls)

    # ---------------- #
    #  To Be Defined   #
    # ---------------- #
    def setup(self, parameters):
        """ """
        self._properties["parameters"] = np.asarray(parameters)

        
    def get_model(self, x, reshapex=True, param=None):
        """ return the model for the given data.
        The modelization is based on legendre polynomes that expect x to be between -1 and 1.
        This will create a reshaped copy of x to scale it between -1 and 1 but
        if x is already as such, save time by setting reshapex to False

        Returns
        -------
        array (size of x)
        """
        if param is not None:
            self.setup(param)
            
        if self.use_legendre:
            if reshapex:
                x = (x-x.min())/(x.max()-x.min()) *2 -1
            
            model = np.asarray([orthogonal.legendre(i)(x) for i in range(self.DEGREE)])
        else:
            model = np.asarray([x**i for i in range(self.DEGREE)])
            
        return np.dot(model.T, self.parameters.T).T
    
    def get_loglikelihood(self, x, y, dy, reshapex=False):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf).

        In the Fitter define _get_model_args_() that should return the input of this
        """
        
        res = y - self.get_model(x, reshapex=reshapex)
        return -0.5 * np.sum(res**2/dy**2)

    # ----------- #
    #  Prior      #
    # ----------- #
    def lnprior(self, parameters):
        """ """
        return 0

    # =============== #
    #  Properties     #
    # =============== #
    @property
    def parameters(self):
        return self._properties["parameters"]

    @property
    def use_legendre(self):
        return self._side_properties["legendre"]
    
    @use_legendre.setter
    def use_legendre(self, uselegendre):
        self._side_properties["legendre"] = bool(uselegendre)

        
###############################
#                             #
# Model Continuum + Gaussian  #
#                             #
###############################
def normal_and_polynomial_model(degree, ngauss):
    """ 
    Build a model with a continuum that has a `degree` polynomial continuum
    and `ngauss` on top of it.
    
    Returns
    -------
    Child of NormPolyModel
    """
    class N_NormPolyModel( NormPolyModel ):
        DEGREE = degree
        NGAUSS = ngauss
        
    return N_NormPolyModel()


class NormPolyModel( PolyModel ):
    DEGREE = 0
    NGAUSS = 0

    PROPERTIES = ["normparameters"]
    
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        
        cls.FREEPARAMETERS = ["a%d"%(i) for i in range(cls.DEGREE)] + \
          ["mu%d"%(i) for i in range(cls.NGAUSS)]  + ["sig%d"%(i) for i in range(cls.NGAUSS)]+ ["ampl%d"%(i) for i in range(cls.NGAUSS)]
        print cls.FREEPARAMETERS
        
        return super(PolyModel,cls).__new__(cls)

    def setup(self, parameters):
        """ read and parse the parameters """
        # good strategy to have 2 names to easily super() the continuum in get_model
        self._properties["parameters"]     = np.asarray(parameters[:self.DEGREE])
        self._properties["normparameters"] = np.asarray(parameters[self.DEGREE:])
        
    def get_model(self, x, reshapex=True, param=None):
        """ return the model for the given data.
        The modelization is based on legendre polynomes that expect x to be between -1 and 1.
        This will create a reshaped copy of x to scale it between -1 and 1 but
        if x is already as such, save time by setting reshapex to False

        Returns
        -------
        array (size of x)
        """
        if param is not None:
            self.setup(param)
        continuum = super(NormPolyModel, self).get_model(x, reshapex=reshapex, param=None)
        return continuum + np.sum([stats.norm.pdf(x, loc=self.normparameters[0+i*3],
                                                scale=self.normparameters[1+i*3])*self.normparameters[2+i*3]
                            for i in range(self.NGAUSS)], axis=0)


    @property
    def normparameters(self):
        return self._properties["normparameters"]
