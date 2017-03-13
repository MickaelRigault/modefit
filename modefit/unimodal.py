#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

import numpy        as np
from scipy          import stats
import matplotlib.pyplot as mpl
# - local dependencies
from .utils import kwargs_update
from .baseobjects import BaseModel,BaseFitter, DataHandler

__all__ = ["normal", 'truncnormal']

# ========================= #
#  Main Methods             #
# ========================= #
def normal(data,errors,names=None,
            masknan=True,**kwargs):
    """ Fit the weighted mean and intrinsic dispersion
    on the data"""
    if masknan:
        flagnan = (data!=data) | (errors !=errors)
        return UnimodalFit(data[~flagnan],errors[~flagnan],
                           names=names[~flagnan] if names is not None else None,**kwargs)
    
    return UnimodalFit(data,errors,names=names,**kwargs)


def truncnormal(data,boundaries,
                errors=None,names=None,
                masknan=True,
                **kwargs):
    """ Fit a truncated normal distribution on the data

    Parameters:
    -----------
    data: [array]
        data following a normal distribution

    boundaries: [float/None, float/None]
        boundaries for the data. Set None for no boundaries

    errors, names: [array, array] -optional-
        error and names of the datapoint, respectively
        
    masknan: [bool] -optional-
        Remove the NaN values entries of the array if True. 

    **kwargs

    Return
    ------
    UnimodalFit
    """
    # ----------
    # - Input 
    if errors is None:
        errors = np.zeros(len(data))
        
    if masknan:
        flagnan = (data!=data) | (errors !=errors)
        fit =  UnimodalFit(data[~flagnan],errors[~flagnan],
                           names=names[~flagnan] if names is not None else None,
                           modelname="TruncNormal",**kwargs)
    else:
        fit =  UnimodalFit(data,errors,
                           names=names if names is not None else None,
                           modelname="TruncNormal",**kwargs)
    fit.model.set_databounds(boundaries)
    return fit
# ========================== #
#                            #
#     Fitter                 #
#                            #
# ========================== #
class UnimodalFit( BaseFitter, DataHandler ):
    """ """
    PROPERTIES         = []
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []

    # =================== #
    #  Initialization    #
    # =================== #
    def __init__(self,data, errors,
                 names=None, use_minuit=True,
                 modelname="Normal"):
        """  low-level class to enable to fit a unimodal model on data
        the given their errors.

        Parameters
        ----------
        data: [array]
            The data that potentially have a bimodal distribution
            (like a step). In case of Cosmological fit, this could be
            the Hubble Residual for instance.
                                   
        errors: [array]
            Errors associated to the data.
                                   
        names: [string-array/None] - optional -
            Names associated with the data. This enable to follow the data
            more easily.
            In Development: If provided, you will soon be able to use
            interactive ploting and see which points corresponds to
            which object.

        use_minuit: [bool] - default True -
            Set the technique used to fit the model to the data.
            Minuit is the iminuit library.
            If not used, the scipy minimisation is used.

        modelname: [string] - deftault Binormal -
            The name of the class used to define the bimodal model.
            This name must be an existing class of this library.

        Return
        -------
        Defines the object
        """
        self.__build__()
        self.set_data(data,errors,names)
            
        # -- for the fit
        # use_minuit has a setter
        self.use_minuit = use_minuit
        self.set_model(eval("Model%s()"%modelname))
        
    # =================== #
    #   Main              #
    # =================== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self,data,errors,names=None):
        """ set the information for the fit.

        Parameters
        ----------
        data: [array]
            The data that potentially have a bimodal distribution
            (like a step). In case of Cosmological fit, this could be
            the Hubble Residual for instance.
                                   
        errors: [array]
            Errors associated to the data.
                                   
        names: [string-array/None] - optional -
            Names associated with the data. This enable to follow the data
            more easily.
            In Development: If provided, you will soon be able to use
            interactive ploting and see which points corresponds to
            which object.

        Returns
        -------
        Void
        """
        # ------------------------ #
        # -- Fatal Input Errors -- #        
        if len(errors)!= len(data):
            raise ValueErrors("data and errors must have the same size")
        
        # -- Warning -- #
        if names is not None and len(names) != len(data):
            warnings.warn("names size does not match the data one. => names ignored")
            names = None

        self._properties["data"] = np.asarray(data)
        self._properties["errors"] = np.asarray(errors)
        self._side_properties["names"] = np.asarray(names) if names is not None else None
        
    def _get_model_args_(self):
        return self.data[self.used_indexes],self.errors[self.used_indexes]

    
    def get_model(self, parameter):
        """ Model distribution (from scipy) estiamted for the
        given parameter. The model dispersion (scale) is the
        square-root quadratic sum of the model's dispersion
        and the median data error

        Return
        ------
        a scipy distribution
        """
        return self.model.get_model(parameter, np.median(self.errors))
        
    def show(self, parameter, ax=None,savefile=None,show=None,
             propmodel={},**kwargs):
        """ show the data and the model for the given parameters
            
        Parameters
        ----------
        parameter: [array]
            Parameters setting the model

        ax: [matplotlib.pyplot Axes] -optional-
            Where the model should be display. if None this
            will create a new figure and a new axes.

        savefile: [string] -optional-
            Save the figure at this location.
            Do not give any extention, this will save a png and a pdf.
            If None, no figure will be saved, but it will be displayed
            (see the show arguement)

        show: [bool] -optional-
            If the figure is not saved, it is displayed except if show
            is set to False
        
        propmodel: [dict] -optional-
            Properties passed to the matplotlib's Axes plot method for the model.
        
        **kwargs goes to matplotlib's hist method

        Return
        ------
        dict (plot information like fig, ax, pl ; output in self._plot)
        """
        from .utils import figout
        # ----------- #
        # - setting - #
        # ----------- #
        import matplotlib.pyplot as mpl
        self._plot = {}
        
        if ax is None:
            fig = mpl.figure(figsize=[8,5])
            ax  = fig.add_axes([0.1,0.1,0.8,0.8])
        elif "plot" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. "+\
                             "No imshow available")
        else:
            fig = ax.figure

        # ------------- #
        # - Prop      - #
        # ------------- #
        defprop = dict(fill=True, fc=mpl.cm.Blues(0.3,0.4), ec=mpl.cm.Blues(1.,1),
                       lw=2, normed=True, histtype="step")
        prop = kwargs_update(defprop,**kwargs)
        # ------------- #
        # - Da Plots  - #
        # ------------- #
        # model range
        datalim   = [self.data.min()-self.errors.max(),
                     self.data.max()+self.errors.max()]
        datarange =datalim[1]-datalim[0]
        x = np.linspace(datalim[0]-datarange*0.1,datalim[1]+datarange*0.1,
                        int(datarange*10))
        # data
        ht = ax.hist(self.data,**prop)
        # model
        prop = kwargs_update(dict(ls="--",color="0.5",lw=2),**propmodel)
        model_ = self.get_model(parameter)
        pl = ax.plot(x, model_.pdf(x),**prop)
        # ------------- #
        # - Output    - #
        # ------------- #
        self._plot["figure"] = fig
        self._plot["ax"]     = ax
        self._plot["hist"]   = ht
        self._plot["model"]  = pl

        fig.figout(savefile=savefile,show=show)
        
        return self._plot

    # =================== #
    #  Properties         #
    # =================== #


# ========================== #
#                            #
#     Model                  #
#                            #
# ========================== #
class ModelNormal( BaseModel ):
    """
    """
    FREEPARAMETERS = ["mean","sigma"]
    
    sigma_boundaries = [0,None]
    
    def setup(self,parameters):
        """ """
        self.mean,self.sigma = parameters


    def get_model(self,parameter, dx):
        """ Scipy Distribution associated to the
        given parameters

        Parameters
        ----------
        parameter: [array]
            Parameters setting the model
        dx: [float]
            Typical representative error on the data. This is to estimate the
            effective dispersion of the gaussian

        Return
        ------
        scipy.norm
        """
        mean, sigma = parameter
        return stats.norm(loc=mean,scale=np.sqrt(sigma**2 + dx**2))
    
    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx, pdf=False):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf) """
        Li = stats.norm.pdf(x,loc=self.mean,scale=np.sqrt(self.sigma**2 + dx**2))
        if pdf:
            return Li
        return np.sum(np.log(Li))
    
    def get_case_likelihood(self,xi,dxi,pi):
        """ return the log likelihood of the given case. See get_loglikelihood """
        return self.get_loglikelihood([xi],[dxi])

    # ----------------------- #
    # - Bayesian methods    - #
    # ----------------------- #
    def lnprior(self,parameter):
        """ so far a flat prior """
        for name_param,p in zip(self.FREEPARAMETERS, parameter):
            if "sigma" in name_param and p<0:
                return -np.inf
        return 0

    # ----------------------- #
    # - Ploting             - #
    # ----------------------- #
    def display(self, ax, xrange, dx, bins=1000,
                ls="--", color="0.4",**kwargs):
        """ Display the model on the given plot.
        This median error is used.
        """
        x = np.linspace(xrange[0],xrange[1],bins)
        lpdf = self.get_loglikelihood(x,np.median(dx), pdf=True)
        
        return ax.plot(x,lpdf,ls=ls, color="k", 
                **kwargs)
        

class ModelTruncNormal( ModelNormal ):
    """ Normal distribution allowing for data boundaries
    (e.g. amplitudes of emission lines are positive gaussian distribution
    """
    
    PROPERTIES = ["databounds"]


    def set_databounds(self,databounds):
        """ boundaries for the data """
        if len(databounds) != 2:
            raise ValueError("databounds must have 2 entries [min,max]")
        self._properties["databounds"] = databounds

    def get_model(self,parameter, dx):
        """ Scipy Distribution associated to the
        given parameters

        Parameters
        ----------
        parameter: [array]
            Parameters setting the model
        dx: [float]
            Typical representative error on the data. This is to estimate the
            effective dispersion of the gaussian

        Return
        ------
        scipy.norm
        """
        mean, sigma = parameter
        tlow,tup = self.get_truncboundaries(dx, mean=mean, sigma=sigma)
        return  stats.truncnorm(tlow,tup,
                        loc=mean, scale=np.sqrt(sigma**2 + dx**2))
    
    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx, pdf=False):
        """ Measure the likelihood to find the data given the model's parameters """
        # info about truncnorm:
        # stats.truncnorm.pdf(x,f0,f1,loc=mu,scale=sigma))
        #  => f0 and f1 are the boundaries in sigma units !
        #  => e.g. stats.truncnorm.pdf(x,-2,3,loc=1,scale=2),
        #     the values below -2sigma and above 3 sigma are truncated

        tlow,tup = self.get_truncboundaries(dx)
        Li = stats.truncnorm.pdf(x,tlow,tup,
                            loc=self.mean, scale=np.sqrt(self.sigma**2 + dx**2))
        if pdf:
            return Li
        return np.sum(np.log(Li))

    def get_truncboundaries(self,dx, mean=None, sigma=None):
        """
        """
        if mean is None:
            mean = self.mean
        if sigma is None:
            sigma= self.sigma
            
        min_ = -np.inf if self.databounds[0] is None else\
            (self.databounds[0]-mean)/np.sqrt(sigma**2 + np.mean(dx)**2)
            
        max_ = +np.inf if self.databounds[1] is None else\
            (mean-self.databounds[1])/np.sqrt(sigma**2 + np.mean(dx)**2)
            
        return min_,max_

    # ----------------------- #
    # - Ploting             - #
    # ----------------------- #
    def display(self, ax, xrange, dx, bins=1000,
                ls="--", color="0.4", show_boundaries=True,
                ls_bounds="-", color_bounds="k", lw_bounds=2,
                **kwargs):
        """ Display the model on the given plot.
        This median error is used.

        The Boundaries are added
        """
        if show_boundaries:
            xlim = np.asarray(ax.get_xlim()).copy()
            if self.databounds[0] is not None:
                ax.axvline(self.databounds[0], ls=ls_bounds,
                           color=color_bounds, lw=lw_bounds)
            if self.databounds[1] is not None:
                ax.axvline(self.databounds[1], ls=ls_bounds,
                           color=color_bounds, lw=lw_bounds)
            ax.set_xlim(*xlim)
        return super(ModelTruncNormal, self).display( ax, xrange, dx, bins=1000,
                                ls=ls, color=color,**kwargs)
        
    # ========================= #
    # = Properties            = #  
    # ========================= # 
    @property
    def databounds(self):
        return self._properties["databounds"]

    @property
    def _truncbounds_lower(self):
        """ truncation boundaries for scipy's truncnorm """
        if self.databounds[0] is None:
            return -np.inf
        return (self.databounds[0]-self.mean)/self.sigma
    
    @property
    def _truncbounds_upper(self):
        """ truncation boundaries for scipy's truncnorm """
        if self.databounds[1] is None:
            return np.inf
        return (self.mean-self.databounds[1])/self.sigma
