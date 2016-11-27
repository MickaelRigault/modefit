#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Basic Filters """

import numpy as np
from scipy.special import orthogonal
from scipy.stats import norm

from .baseobjects import BaseModel, BaseFitter, DataHandler

__all__ = ["get_polyfit","get_polygaussfit"]


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


def get_polygaussfit(x, y, dy, degree,  nbr_gauss, **kwargs):
    """ Get the object to fit the sum of a `degree`th order polynomes and a number of normal functions.

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

    nbr_gauss: 
        Number of gaussian components to add.
        Note that unless parameter limits are given in the fit, 
        nbr_gauss>1 will be degenerate.

    Returns
    -------
    PolyGaussFit
    Order of fit parameters: degree polynomial coefficients, in increasing order,
    subsequently (position, sigma and amplitude) of each gaussian.

    """
    return PolyGaussFit(np.asarray(x), np.asarray(y),
                        np.asarray(dy), degree, nbr_gauss,
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
             show_model=True,
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
        prop = kwargs_update( dict(ms=15, mfc=mpl.cm.Blues(0.6,0.5), mec=mpl.cm.Blues(0.8,0.9),
                                   ls="None",mew=1.5, marker="o", zorder=5), **kwargs)
        
        pl = ax.plot(self.xdata,self.data, **prop)
        er = ax.errorscatter(self.xdata,self.data, dy=self.errors, zorder=prop["zorder"]-1,
                             ecolor=ecolor)
        # - Model
        if show_model:
            xx = np.linspace(self.xdata.min()-np.abs(self.xdata.max()), self.xdata.max()*2, 1000)
            
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




####################################
#                                  #
#                                  #
#    Polynomial + Gaussian Fits    #
#                                  #
#                                  #
####################################
class PolyGaussFit( BaseFitter, DataHandler ):
    """ """
    PROPERTIES         = ["xdata"]
    DERIVED_PROPERTIES = []

    def __init__(self, x, y, dy, degree, nbr_gauss,
                 names=None):
        """ """
        self.set_data(x, y, dy)
        self.set_model(polygauss_model(degree,nbr_gauss))
        
    # ============== #
    #  Main Methods  #
    # ============== #
    # - To Be Defined
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        return self.xdata, self.data, self.errors
    
    # - Super It
    def set_data(self, x, y, dy=None, names=None):
        """ Basic method to set the data """
        self._properties["xdata"]  = x
        super(PolyGaussFit, self).set_data(y, errors=dy, names=names)

    def set_model(self, model, **kwargs):
        """ """
        super(PolyGaussFit, self).set_model(model, **kwargs)


    def show(self, savefile=None, show=True, ax=None,
             show_model=True,
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
        prop = kwargs_update( dict(ms=15, mfc=mpl.cm.Blues(0.6,0.5), mec=mpl.cm.Blues(0.8,0.9),
                                   ls="None",mew=1.5, marker="o", zorder=5), **kwargs)
        
        pl = ax.plot(self.xdata,self.data, **prop)
        er = ax.errorscatter(self.xdata,self.data, dy=self.errors, zorder=prop["zorder"]-1,
                             ecolor=ecolor)
        # - Model
        if show_model:
            xx = np.linspace(self.xdata.min()-np.abs(self.xdata.max()), self.xdata.max()*2, 1000)
            
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
    
    


############################
#                          #
# Model Polynom + gauss's  #
#                          #
############################

def polygauss_model(degree,nbr_gauss):
    """ 
    On the fly creation of model with the required fixed parameters
    Returns
    -------
    Child of PolyGaussModel 
    """
    class N_PolyGaussModel( PolyGaussModel ):
        DEGREE = degree
        NBRGAUSS = nbr_gauss

    return N_PolyGaussModel()


class PolyGaussModel( BaseModel ):
    """ Virtual Class able to handle sum any polynomial order fitter and any number of gaussians"""
    DEGREE = 0
    NBRGAUSS = 0
    
    PROPERTIES         = ["parameters"]
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []

    # ================ #
    #  Main Method     #
    # ================ #
    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of models """
        
        cls.FREEPARAMETERS = ["a%d"%(i) for i in range(cls.DEGREE+3*cls.NBRGAUSS)]
        return super(PolyGaussModel,cls).__new__(cls)

    # ---------------- #
    #  To Be Defined   #
    # ---------------- #
    def setup(self, parameters):
        """ """
        self._properties["parameters"] = np.asarray(parameters)

        
    def get_model(self, x, param=None):
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

        # Polynomial part
        xpoly = np.asarray([x**i for i in range(self.DEGREE)])
        imodel = np.dot(xpoly.T,self.parameters[0:self.DEGREE].T).T
        # Gauss part
        for i in range(self.NBRGAUSS):
#            rv = norm(loc = self.parameters[self.DEGREE+i*3], scale = self.parameters[self.DEGREE+i*3+1])
#            imodel += rv.pdf(x) * self.parameters[self.DEGREE+i*3+2]
            imodel += self.parameters[self.DEGREE+i*3+2] * np.exp( - (x-self.parameters[self.DEGREE+i*3])**2/(2*self.parameters[self.DEGREE+i*3+1]**2) )
        
        return imodel
    
    def get_loglikelihood(self, x, y, dy):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf).

        In the Fitter define _get_model_args_() that should return the input of this
        """
        
        res = y - self.get_model(x)
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

