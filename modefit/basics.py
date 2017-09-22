#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Basic Filters """
import warnings
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
        self.model.set_xsource(x)
        
    # ============== #
    #  Main Methods  #
    # ============== #
    # - To Be Defined
    def _get_model_args_(self):
        """ see model.get_loglikelihood"""
        return self.data, self.errors
    
    # - Super It
    def set_data(self, x, y, dy=None, names=None):
        """ Basic method to set the data """
        self._properties["xdata"]  = x
        self._derived_properties["xscaled"]  = (x-np.min(x))/(np.max(x)-np.min(x))*2-1.
        super(PolynomeFit, self).set_data(y, errors=dy, names=names)

    def set_model(self, model, use_legendre=True, **kwargs):
        """ """
        super(PolynomeFit, self).set_model(model, **kwargs)
        self.model.use_legendre=use_legendre

    def _display_data_(self, ax, ecolor="0.3", **prop):
        """ """
        from .utils import errorscatter
        pl = ax.plot(self.xdata,self.data, **prop)
        er = ax.errorscatter(self.xdata,self.data, dy=self.errors, zorder=prop["zorder"]-1,
                             ecolor=ecolor)
        return pl
        
    def show(self, savefile=None, show=True, ax=None,
             show_model=True, xrange=None, parameters=None,
             mcmc=False, nsample=100, mlw=2, ecolor="0.3",
             mcmccolor=None, modelcolor= "k", modellw=2, 
             **kwargs):
        """ """
        import matplotlib.pyplot as mpl 
        from .utils import figout, kwargs_update
        
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
        pl = self._display_data_(ax, ecolor=ecolor,**prop)
        
        # - Model
        if show_model and (self.has_fit_run() or parameters is not None): 
            if xrange is not None:
                xx = np.linspace(self.xdata.min(), self.xdata.max(), 1000)
                print(xx)
                self.model.set_xsource(xx)
                
            if not mcmc or (parameters is not None):
                model_to_show = self.model.get_model() if parameters is None else self.get_model(np.asarray(parameters))
                    
                model = ax.plot(self.model.xsource, model_to_show, ls="-", lw=modellw,
                                color=modelcolor, scalex=False, scaley=False, zorder=np.max([prop["zorder"]-2,1]))
                
            elif not self.has_mcmc():
                warnings.warn("No MCMC loaded. use run_mcmc()")
                model = []
            else:
                if mcmccolor is None:
                    mcmccolor = mpl.cm.binary(0.6,0.3)
                model = [ax.plot(xx,self.model.get_model(param=param), color=mcmccolor,
                                scalex=False, scaley=False, zorder=np.max([prop["zorder"]-3,1]))
                        for param in self.mcmc.samples[np.random.randint(len(self.mcmc.samples), size=nsample)]]
                
                model.append(ax.plot(xx,self.model.get_model(param=np.asarray(self.mcmc.derived_values).T[0]), 
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
        self.__build__()
        self.set_data(x, y, dy)
        self.set_model(normal_and_polynomial_model(degree, ngauss),
                           use_legendre=legendre)
        self.model.set_xsource(x)
        
    def _display_data_(self, ax, ecolor="0.3", **prop):
        """ """
        from .utils import specplot
        return ax.specplot(self.xdata,self.data, var=self.errors**2,
                        bandprop={"color":ecolor},**prop)
        
    def show(self,savefile=None, show=True, ax=None, show_model=True, xrange=None,
                 show_gaussian=False,
                 mcmc=False, nsample=100, mlw=2, ecolor='0.3',
                 mcmccolor=None, modelcolor='k', modellw=2, **kwargs):
        """ """
        import matplotlib.pyplot as mpl 
        from .utils import figout, errorscatter, kwargs_update

        pkwargs = kwargs_update(dict(ls="-", marker="None"),**kwargs)
        
        pl = super(NormPolynomeFit, self).show(savefile=None, show=False, ax=ax,
                                          show_model=show_model, xrange=xrange,
                                          mcmc=mcmc, nsample=nsample, mlw=mlw,
                                          ecolor=ecolor,
                                          mcmccolor=mcmccolor, modelcolor=modelcolor,
                                          modellw=modellw, **pkwargs)
        
        # -- Add individual gaussian
        if show_gaussian:
            cont = self.model._get_continuum_()
            [pl["ax"].plot(self.model.xsource,
                               self.model.get_ith_gaussian(self.model.xsource,i, add_continuum=False)+cont,
                               ls="-", lw=modellw/2.,alpha=0.5,
                               color=modelcolor, scalex=False, scaley=False, zorder=np.max([pl["prop"]["zorder"]-1,1]))
                               
            for i in range(self.model.NGAUSS)]

        pl["figure"].figout(savefile=savefile, show=show)
        return pl
    
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
    
    PROPERTIES         = ["parameters",
                          "xsource","xsource_start","xsource_steps"]
    SIDE_PROPERTIES    = ["legendre"]
    DERIVED_PROPERTIES = ["xsource_scaled"]

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

        
    def get_model(self, x=None, param=None):
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
            
        if x is not None:
            self.set_xsource(x)
            
        if self.use_legendre:            
            model = np.asarray([orthogonal.legendre(i)(self.xsource_scaled) for i in range(self.DEGREE)])
        else:
            model = np.asarray([self.xfit**i for i in range(self.DEGREE)])
            
        return np.dot(model.T, self.parameters.T).T
    
    def get_loglikelihood(self, y, dy, x=None):
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

    @property
    def use_legendre(self):
        return self._side_properties["legendre"]
    
    @use_legendre.setter
    def use_legendre(self, uselegendre):
        self._side_properties["legendre"] = bool(uselegendre)

    # -------------
    # x values
    def set_xsource(self, x):
        self._properties["xsource_start"],self._properties["xsource_steps"] = self.parse_xdata(np.asarray(x))
        self._properties["xsource"]        = np.asarray(x)
        self._derived_properties["nsteps"] = len(self.xsource_steps)
        self._derived_properties["xsource_scaled"] = None
        
    def parse_xdata(self, x):
        """ converts the given array in steps+star format """
        try:
            return x[0],x[1:]-x[:-1]
        except:
            raise ValueError('Incorrect x format', x)
    
    @property
    def xsource(self):
        """ """
        return self._properties["xsource"]

    @property
    def xfit(self):
        return self.xsource
    
    @property
    def xsource_scaled(self):
        """ """
        if self._derived_properties["xsource_scaled"] is None and self.xsource is not None:
            self._derived_properties["xsource_scaled"] = (np.asarray(self.xsource, dtype="float")-self.xsource.min())/(self.xsource.max()-self.xsource.min()) *2 -1
        return self._derived_properties["xsource_scaled"]
            
    @property
    def xsource_start(self):
        """ x-pixelisation upon which the lbda is built. 'start, step' format """
        return self._properties["xsource_start"]
    
    @property
    def xsource_steps(self):
        """ x-pixelisation upon which the lbda is built. 'start, step' format """
        return self._properties["xsource_steps"]

    # - pixels Statistics
    @property
    def nsteps(self):
        """ Size of the step array """
        self._derived_properties["nsteps"]
        
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
        if not hasattr(cls,"FREEPARAMETERS"):
            cls.FREEPARAMETERS = ["a%d"%(i) for i in range(cls.DEGREE)]
        else:
            cls.FREEPARAMETERS += ["a%d"%(i) for i in range(cls.DEGREE)]
        cls.FREEPARAMETERS += ["mu%d"%(i) for i in range(cls.NGAUSS)]  + ["sig%d"%(i) for i in range(cls.NGAUSS)]+ ["ampl%d"%(i) for i in range(cls.NGAUSS)]
        
        return super(PolyModel,cls).__new__(cls)

    def setup(self, parameters):
        """ read and parse the parameters """
        # good strategy to have 2 names to easily super() the continuum in get_model
        self._properties["parameters"]     = np.asarray(parameters[:self.DEGREE])
        self._properties["normparameters"] = np.asarray(parameters[self.DEGREE:])


    def get_ith_gaussian(self, x, ithgauss, add_continuum=False, param=None):
        """ """
        if param is not None:
           self.setup(param)
        cont = 0 if not add_continuum else self._get_continuum_(x)
        return stats.norm.pdf(x, loc=self.normparameters[ithgauss::self.NGAUSS][0],
                            scale=self.normparameters[ithgauss::self.NGAUSS][1])*self.normparameters[ithgauss::self.NGAUSS][2] + cont

    def get_model(self, x=None, param=None):
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
        if x is not None:
            self.set_xsource(x)
            
        continuum = self._get_continuum_()
        
        return continuum + np.sum([stats.norm.pdf(self.xfit, loc=self.normparameters[i::self.NGAUSS][0],
                                                scale=self.normparameters[i::self.NGAUSS][1])*self.normparameters[i::self.NGAUSS][2]
                            for i in range(self.NGAUSS)], axis=0)

    def _get_continuum_(self,x=None):
        """ """
        return super(NormPolyModel, self).get_model(x,param=None)
    
    @property
    def normparameters(self):
        return self._properties["normparameters"]
