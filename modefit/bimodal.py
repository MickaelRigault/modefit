#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

import numpy        as np
from scipy          import stats
import matplotlib.pyplot as mpl
import warnings
# - local dependencies
from .utils import kwargs_update
from .baseobjects import BaseModel,BaseFitter, DataHandler
from .unimodal import normal

__all__ = ["stepfit","bimodal_fit"]

# ========================= #
#  Main Methods             #
# ========================= #
def bimodal_fit(data, errors, proba=None, masknan=True, names=None, **kwargs):
    """  Fit a binormal distribution in your data.

    
    Parameters
    -----------
    data, errors: [array, array]
        The y-axis data (and its errors) that potentially is bimodal.

    proba: [array] -optional-
        If you already know the probability of each data point to be in 
        one mode or the other, set it here. It is fit otherwise.

    masknan: [bool] -optional-
        Remove the nan values (in x, data or errors) prior to load StepFit

    names: [array] -optional-
        names for the data points.

    Returns
    -------
    BimodalFit
    """
    if masknan:
        flagnan = (data !=data) | (errors != errors)
        
        return BimodalFit(data[~flagnan],errors[~flagnan],
                          proba=proba[~flagnan]   if proba is not None else None,
                          names = np.asarray(names)[~flagnan] if names is not None else None,
                          modelname="Binormal" if proba is not None else "FloatingBinormal",
                         **kwargs)

    return BimodalFit(data,errors,proba=proba,
                          names = names,
                          modelname="Binormal" if proba is not None else "FloatingBinormal",
                         **kwargs)


def stepfit(x,data,errors,proba=None,dx=None,xcut=None,
            masknan=True, names=None, **kwargs):
    """ Fit a Step in you Data !
    This function will return a StepFit object that will allow you
    to fit a step in you data and easily analysis this step.

    Parameters
    ----------
    x: [array]
        The x-axis of you data that will be used to define which point is
        above or below the given xcut value.
        If proba if provided, this will only be used for plotting.

    data, errors: [array, array]
        The y-axis data (and its errors) that potentially is bimodal.


    proba: [array] -optional-
        If you already know the probability for each point to be below (0) or
        above (1) the step, them give it here.
    
    dx: [array] -optional-
        If proba is None, dx (the error on the x-axis) will enable to measure the
        probability to be above or below the step (at x=xcut).
        If proba is None and dx is None, the proba will be ones and zeros.

    xcut: [float] -optional / required if proba is None -
        Define where the step is located.
        If proba is given, this will only be used for plotting

    masknan: [bool] -optional-
        Remove the nan values (in x, data or errors) prior to load StepFit

    names: [array] -optional-
        names for the data points.

    Returns
    -------
    StepFit.
    """
    if masknan:
        flagnan = (x!=x) | (data !=data) | (errors != errors)
        
        return StepFit(x[~flagnan],data[~flagnan],errors[~flagnan],
                       proba=proba[~flagnan]   if proba is not None else None,
                       dx=dx[~flagnan]         if dx    is not None else None,
                       names = np.asarray(names)[~flagnan] if names is not None else None,
                       xcut=xcut,**kwargs)
    
    return StepFit(x,data,errors,
                   proba=proba,dx=dx,xcut=xcut,
                   **kwargs)


# ========================== #
#                            #
#     Fitter                 #
#                            #
# ========================== #
class BimodalFit( BaseFitter, DataHandler ):
    """ """

    PROPERTIES         = ["proba"]
    SIDE_PROPERTIES    = ["unimodal"]
    DERIVED_PROPERTIES = []
    
    # ========================= #
    # = Initialization        = #  
    # ========================= #
    def __init__(self, data, errors, proba=None,
                 names=None, use_minuit=True,
                 modelname="Binormal", empty=False):
        """  low-level class to enable to fit a bimodal model on data
        given a probability of each point to belong to a group or the other.

        Parameters
        ----------
        data: [array]
            The data that potentially have a bimodal distribution
            (like a step). In case of Cosmological fit, this could be
            the Hubble Residual for instance.
                                   
        errors: [array]
            Errors associated to the data.
                                   
        proba: [array] -optional-
            Probability of the data to belong to one mode or the other.
            The probabilies must be float between 0 and 1.
            If Not provided, the modelname should be 'ModelFloatingBinormal'
    
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
        if empty:
            return
        
        self.set_data(data,errors,proba,names)
            
        # -- for the fit
        # use_minuit has a setter
        self.use_minuit = use_minuit
        self.set_model(eval("Model%s()"%modelname))

    def set_data(self,data,errors,proba=None,names=None):
        """ set the information for the fit.

        Parameters
        ----------
        data: [array]
            The data that potentially have a bimodal distribution
            (like a step). In case of Cosmological fit, this could be
            the Hubble Residual for instance.
                                   
        errors: [array]
            Errors associated to the data.
                                   
        proba: [array]
            Probability of the data to belong to one mode or the other.
            The probabilies must be float between 0 and 1
    
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
        if proba is not None and ((np.asarray(proba)>1).any() or (np.asarray(proba)<0).any()):
            raise ValueError("probabilities (proba) must be between 0 and 1")
        
        if (proba is not None and len(proba)!=len(data)) or len(errors)!= len(data):
            raise ValueError("data, errors and proba must have the same size")
        
        # -- Warning -- #
        if names is not None and len(names) != len(data):
            warnings.warn("names size does not match the data one. => names ignored")
            names = None

        self._properties["data"] = np.asarray(data)
        self._properties["errors"] = np.asarray(errors)
        self._properties["proba"] = np.asarray(proba) if proba is not None else proba

        self._side_properties["names"] = np.asarray(names) if names is not None else None
        
        
    # ========================= #
    # = Fit                   = #  
    # ========================= #        
    def _get_model_args_(self):
        if ModelFloatingBinormal in self.model.__class__.__mro__:
            return self.data[self.used_indexes],self.errors[self.used_indexes]
        return self.data[self.used_indexes],self.errors[self.used_indexes],self.proba[self.used_indexes]
        
    # ====================== #
    # Properties             #
    # ====================== #

    @property
    def proba(self):
        """ probability to belog to the first group """
        return self._properties["proba"]
    
    @property
    def proba_r(self):
        """ probability to belog to the second group (1-proba)"""
        return 1-self.proba

    @property
    def unimodal(self):
        """ the unimodel class that could be used for comparison """
        return self._side_properties["unimodal"]

    def set_unimodal(self, runfit=True):
        """ will load a normal distribution fit with the same data """
        self._side_properties["unimodal"] = normal(self.data, self.errors, names=self.names)
        if runfit:
            self.unimodal.fit()
            
# ========================== #
#                            #
#     The Model              #
#                            #
# ========================== #
class ModelBinormal( BaseModel ):
    """ Model for a Bi Normal Distribution (2 Gaussians) """

    SIDE_PROPERTIES = []
    
    FREEPARAMETERS = ["mean_a","sigma_a",
                      "mean_b","sigma_b"]
    # -------------------
    # - Initial Guesses    
    sigma_a_guess = 0
    sigma_a_fixed = False
    sigma_a_boundaries = [0,None]
    
    sigma_b_guess = 0
    sigma_b_fixed = False
    sigma_b_boundaries = [0,None]

    def setup(self,parameters):
        """ """
        self.mean_a,self.sigma_a,self.mean_b,self.sigma_b = parameters

    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx,p):
        """ Measure the likelihood to find the data given the model's parameters """
        return np.sum(np.log( self.pdf(x,dx,p) ))

    # ------------- #
    # - Modeling  - #
    # ------------- #
    def cdf(self, x, dx, p):
        """ """
        return p * stats.norm.cdf(x,loc=self.mean_a,scale=np.sqrt(self.sigma_a**2 + dx**2)) + \
               (1-p) * stats.norm.cdf(x,loc=self.mean_b,scale=np.sqrt(self.sigma_b**2 + dx**2))
               
    def pdf(self,x, dx, p):
        """ return the log likelihood of the given case. See get_loglikelihood """
        return p * stats.norm.pdf(x,loc=self.mean_a,scale=np.sqrt(self.sigma_a**2 + dx**2)) + \
               (1-p) * stats.norm.pdf(x,loc=self.mean_b,scale=np.sqrt(self.sigma_b**2 + dx**2))

    def get_chauvenet_mask(self, x, dx, p, outlier_cut=0.5):
        """
        outlier_proba: [None, 0<float<1] -optional-
        If you perform an outlier rejection following the Chauvenet's criterium,
        if a target as less than `outlier_proba` chance to exist, it is removed.
        This account for the size of you sample and assumes a 2-tailed outlier
        rejection.
        To work, the distribution have to be gaussian. This criterium is not iterative
        (i.e. only applied once).
            
        The Historical Chauvenet's criterium uses 'outlier_proba'=0.5
            
        Example:
            For 100 points, with 'outlier_proba'=0.5 (historical Chauvenet choice)
            the outlier rejection consist of a 2.8sigma clipping.
            With 'outlier_proba'=0.1, this becomes a 3.3 sigma cliping.    
        """
        cdf = self.cdf(x, dx, p )
        outlier_cut = outlier_cut/(2.*len(x))
        return (cdf<outlier_cut) + (cdf>(1-outlier_cut))
    
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
    # - Model Particularity - #
    # ----------------------- #
    def _minuit_chi2_(self,mean_a,sigma_a,
                     mean_b,sigma_b):
        """
        """
        parameter = mean_a,sigma_a,mean_b,sigma_b
        return self.get_chi2(parameter)


    # ================== #
    #   Properties       #
    # ================== #


# ========================== #
#                            #
#   Fit the proba as well    #
#                            #
# ========================== #
class ModelFloatingBinormal( ModelBinormal ):
    """ """
    FREEPARAMETERS = ["mean_a","sigma_a",
                      "mean_b","sigma_b","proba_a"]
        
    proba_a_guess = 0.5
    proba_a_boundaries=[0.00000001,0.99999999]
    
    def setup(self,parameters):
        """ """
        self.mean_a,self.sigma_a,self.mean_b,self.sigma_b, self.proba_a = parameters

    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx):
        """ Measure the likelihood to find the data given the model's parameters """
        return np.sum(np.log( self.pdf(x,dx) ))

    # ------------- #
    # - Modeling  - #
    # ------------- #
    def cdf(self, x, dx):
        """ """
        return self.proba_a * stats.norm.cdf(x,loc=self.mean_a,scale=np.sqrt(self.sigma_a**2 + dx**2)) + \
               (1-self.proba_a) * stats.norm.cdf(x,loc=self.mean_b,scale=np.sqrt(self.sigma_b**2 + dx**2))
               
    def pdf(self,x, dx):
        """ return the log likelihood of the given case. See get_loglikelihood """
        return self.proba_a * stats.norm.pdf(x,loc=self.mean_a,scale=np.sqrt(self.sigma_a**2 + dx**2)) + \
               (1-self.proba_a) * stats.norm.pdf(x,loc=self.mean_b,scale=np.sqrt(self.sigma_b**2 + dx**2))

    def get_chauvenet_mask(self, x, dx, outlier_cut=0.5):
        """
        outlier_proba: [None, 0<float<1] -optional-
        If you perform an outlier rejection following the Chauvenet's criterium,
        if a target as less than `outlier_proba` chance to exist, it is removed.
        This account for the size of you sample and assumes a 2-tailed outlier
        rejection.
        To work, the distribution have to be gaussian. This criterium is not iterative
        (i.e. only applied once).
            
        The Historical Chauvenet's criterium uses 'outlier_proba'=0.5
            
        Example:
            For 100 points, with 'outlier_proba'=0.5 (historical Chauvenet choice)
            the outlier rejection consist of a 2.8sigma clipping.
            With 'outlier_proba'=0.1, this becomes a 3.3 sigma cliping.    
        """
        cdf = self.cdf(x, dx )
        outlier_cut = outlier_cut/(2.*len(x))
        return (cdf<outlier_cut) + (cdf>(1-outlier_cut))

    def _minuit_chi2_(self,mean_a,sigma_a,
                     mean_b,sigma_b, proba_a):
        """
        """
        # USELESS I THINK
        parameter = mean_a,sigma_a,mean_b,sigma_b, proba_a
        return self.get_chi2(parameter)

# ========================== #
#                            #
#   Step                     #
#                            #
# ========================== #
class StepFit( BimodalFit ):
    """
    """
    PROPERTIES = ["x","xcut"]
    SIDE_PROPERTIES = ["dx"]
    
    def __init__(self,x,data,errors,
                 proba=None,dx=None,
                 xcut=None,name=None,
                 use_minuit=None,
                 empty=False, **kwargs):
        """ the ability to use an extra x-axis value to potentially define *proba* if needed
          and to show a *x,data* plot taking into account this probability.
          See further details in ProbaFit_library.BimodalFit
        =
        x: [float-array]
            The x-axis where the cut is made between the 2 populations.
            If *proba* is not pre-defined, this will be used together
            with xcut to do it. This must have the same size as *data*

        data: [float-array]
            The data that potentially have a bimodal distribution (like a step).
                                   
        errors: [float-array]
            Errors associated to the data.
                              
        proba: [float-array or None] - optional -
            (either proba or xcut must be given)
            Probability of the data to belong to one mode or the other.
            If None, x and xcut (and dx if any) will be used to defined
            proba.
            If not None, proba must be privided. The probabilies must
            be float between 0 and 1.
                                   
        name: [string-array or None] - optional -
            name of the data points. 

        
        dx: [float-array or None] - optional -
            errors on the xaxis. these errors will be used if proba have to
            be defined using x and xcut.
            This will assume symetric gaussian errors. (See self.get_proba).
        
        xcut: [float or None] - optional -
            (either proba or xcut must be given)
            This is were the split is between the 2 populations.
            This value will be used if proba have to be defined based on this and
            the x and dx (if any).

        **kwargs goes to the BimodalFit fit (e.g. use_minuit, modelname)

        Return
        -------
        Void, defines the object
        """
        # -- Init Tests -- #
        if len(x) != len(data):
            raise ValueErrors("x and data must have the sample size (%d vs. %d)"%(len(x),len(data)))
        
        if dx is not None and len(x) != len(dx):
            raise ValueErrors("x and dx must have the sample size (%d vs. %d). Give dx=None not to use it"%(len(x),len(dx)))

        # ----------------------- #
        # -- The x Information -- #
        # ----------------------- #
        # -- basic x-stuffs
        self._properties["x"]       = np.asarray(x)
        self._side_properties["dx"] = dx if dx is None else np.asarray(dx)
        self._properties["xcut"]    = xcut
        # -- The Probability given the step location
        if proba is None:
            if xcut is None:
                raise ValueError("You need to give either proba or xcut to enable to define proba")
            proba = self.get_proba()
                    
        # ----------------------- #
        # -- The Mother's Init -- #
        # ----------------------- #
        super(StepFit,self).__init__(data, errors, proba,
                                     **kwargs)
        

    # ========================= #
    # = Step Stuffs           = #  
    # ========================= #
    def get_proba(self,xcut=None):
        """
         This function will split the sample in two at the given *xcut* value.
         If no *dx* provided, *proba* will be 0 or 1. Otherwise errors
         on the x-axis will be used to define non-trivial probability.
         CAUTION: if there is errors, this assumes then symetric and gaussian.

        Parameters:
        -----------

        xcut: [float/None]         The x-value where the sample is splitted in 2.
                                   If None is set, self.xcut will be used if it
                                   already is defined. This will update self.xcut.

        Returns:
        --------
        array of float (between 0 and 1 ;size of x)
        """
        if xcut is not None:
            self._side_properties['xcut'] = xcut
        
        if self.dx is None: # - faster this way
            return np.asarray([0 if x>self.xcut else 1 for x in self.x])
        
        return np.asarray([stats.norm(loc=x,scale=dx).cdf(self.xcut)
                         for x,dx in zip(self.x,self.dx)])
    
    # ========================= #
    # = Step Shows            = #  
    # ========================= #
    def show(self,savefile=None,axes=None, #rangey=[-0.6,0.6],
             figure=None,cmap=mpl.cm.viridis, ybihist=True,
             propaxes={}, rangex=None,rangey=None,
             show_xhist=False,
             binsx=10,binsy=10,**kwargs):
        """ Plot x, data in a 3-axes plot

        Parameters:
        -----------
        savefile: [string/None]
            If you wish to save the plot in the given filename (*without extention*).
            This will create a `savefile`.pdf and `savefile`.png
            The plot won't be show if it is saved.

        axes: [3-array axes/None]
            Give mpl axes in which the plot will be drawn 3 axes must be given
            (axsc, axhistx, axhisty). The latest 2 (both or only one) could be set 
            to None if you do not wish to have the corresponding histogram plotted.
                                    
        figure: [mpl.Figure]
            If you did not provided axes, you can give a figure into which the axes
            will be drawn, otherwise this this create a new one.
            
        ybihist: [bool]
           Oppose the two y-histogram
        
        propaxes: [dict]
           properties entering the 'add_threeaxes' as kwargs.
        
        **kwargs   goes to matplotlib's Axes.scatter

        Returns:
        --------
        Void
        """
        

        # =================
        # Setting the Axes
        # =================
        if axes is not None:
            if len(axes) != 3:
                raise ValueErrors("the input 'axes' must be a 3d-array"+\
                                  " (ax,axhistx,axhisty)")
            ax,axhistx,axhisty = axes
            fig = ax.figure
        else:
            from .utils import add_threeaxes
            fig = figure if figure is not None else \
              mpl.figure(figsize=[7,5]) if show_xhist else mpl.figure(figsize=[8,4])
            ax,axhistx,axhisty = fig.add_threeaxes(xhist=show_xhist,**propaxes)

        # =================
        # The Scatter Plot
        # =================
        ecolor = kwargs.pop("ecolor","0.7")
        ax.errorbar(self.x,self.data,xerr=self.dx,yerr=self.errors,
                    ecolor=ecolor,ls="None", marker=None,label="_no_legend_",
                    zorder=2)

        prop = kwargs_update({"s":150,"edgecolors":"0.7","linewidths":1,"zorder":5,},
                             **kwargs)
        ax.scatter(self.x,self.data,c=self.proba,cmap=cmap,
                   **prop)
        if self.xcut is not None:
            ax.axvline(self.xcut,ls="--",color="k",alpha=0.8)

        # =================
        # The Histograms
        # =================
        propa = {"histtype":"bar","fc":cmap(0.95),"ec":ecolor,"fill":True}
        propb = {"histtype":"bar","fc":cmap(0.05),"ec":ecolor,"fill":True}
        
        # - x-hist
        if axhistx is not None:
            axhistx.hist(self.x,weights=1-self.proba,
                         range=rangex,bins=binsx,**propb)
            axhistx.hist(self.x,weights=self.proba,
                         range=rangex,bins=binsx,**propa)
        # - y-hist
        if axhisty is not None:
            axhisty.hist(self.data,weights=1-self.proba,orientation="horizontal",
                         range=rangey,bins=binsy,**propb)
            axhisty.hist(self.data,weights=self.proba*(-1 if ybihist else 1),
                         orientation="horizontal",range=rangey,bins=binsy,
                         **propa)
            
            if ybihist:
                axhisty.set_xlim(-axhisty.get_xlim()[-1]*1.2,axhisty.get_xlim()[-1]*1.2)
                
        # =================
        # The Fitted Values
        # =================
        #-- if you already made the fit
        if self.has_fit_run():
            # -- To be improve, this does not move with the axis if user does so.
            from .utils import hline,hspan
            # => Folded
            if "__iter__" in dir(self.fitvalues['mean_a']):
                mean_a, mean_aerr,mean_b, mean_berr = \
                      np.mean(self.fitvalues['mean_a']),np.mean(self.fitvalues['mean_a.err']), \
                      np.mean(self.fitvalues['mean_b']),np.mean(self.fitvalues['mean_b.err'])
            # => Non-folded
            else:
                mean_a, mean_aerr,mean_b, mean_berr = \
                      self.fitvalues['mean_a'],self.fitvalues['mean_a.err'], \
                      self.fitvalues['mean_b'],self.fitvalues['mean_b.err']

            line_prop = dict(zorder=4)
            for ax_,cut in [[ax,self.xcut],[axhisty,0]]:
                if ax_ is None: continue
                
                ax_.hline(mean_a,xmax=cut,
                       color=cmap(0.95),alpha=0.8, **line_prop)
                ax_.hspan(mean_a-mean_aerr,mean_a+mean_aerr,
                        xmax=cut,edgecolor="0.7",
                        color=cmap(0.95,0.2),**line_prop)
            
                ax_.hline(mean_b,xmin=cut,
                     color=cmap(0.05,0.8),**line_prop)
                ax_.hspan(mean_b-mean_berr,mean_b+mean_berr,
                        xmin=cut,edgecolor="0.7",
                        color=cmap(0.05,0.2),**line_prop)
                
        # ---------
        # - Plots
        self._plot = {}
        self._plot["ax"] = [ax,axhistx,axhisty]
        self._plot["fig"] = fig
        return self._plot


    def show_kfolding(self, savefile=None, ax=None, lc="k", lprop={},
                      show=True,show_legend=True,**kwargs):
        """ """
        if not self.has_kfold():
            raise AttributeError("No kfolding set.")

        from .utils import figout
        self._plot = {}
        
        # - Settings - #
        if ax is None:
            fig = mpl.figure(figsize=[8,6])
            ax  = fig.add_axes([0.16,0.16,0.73,0.73])
            ax.set_xlabel(r"$\mathrm{Step\ significance\ [\sigma]}$" ,fontsize = "xx-large")
            ax.set_ylabel(r"$\mathrm{Frequency}$",fontsize = "xx-large")
        elif "imshow" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. "+\
                             "No imshow available")
        else:
            fig = ax.figure

        # -  Prop   - #
        histprop = kwargs_update(
            dict(bins=20, fill=True, histtype="step",
                 lw=3, ec=mpl.cm.Blues(0.8),fc=mpl.cm.Blues(0.6,0.4),
                label=r"$\mathrm{K\,folded\ detection}$"), **kwargs)
        
        # - Da Plot - #
        ax.hist(self.kfold.modelstep[0]/self.kfold.modelstep[1], **histprop)

        ax.axvline(self.modelstep[0]/self.modelstep[1] * np.sqrt(len(self.kfold.fitvalues['id'][0])/float(self.npoints)), 
                color=lc, label=r"$\mathrm{Expected\ K\,folded\ detection}$",
                **lprop)
        
        ax.axvline(self.modelstep[0]/self.modelstep[1], ls ="--",color=lc,
                label=r"$\mathrm{Analysis\ detection}$",
                **lprop)
        # -- out
        if show_legend:
            ax.legend(loc="best", fontsize="large", frameon=False)

        fig.figout(savefile=savefile, show=show)



        
    # ========================= #
    # = Properties            = #  
    # ========================= #
    @property
    def modelstep(self):
        """ The diffence of the means of the two modes, and its associated error """
        return self.fitvalues['mean_a'] - self.fitvalues["mean_b"],\
          np.sqrt(self.fitvalues["mean_a.err"]**2 + self.fitvalues["mean_b.err"]**2)

    @property
    def x(self):
        """ x-axis of the data """
        return self._properties["x"]
    @property
    def dx(self):
        """ error along the x-axis of the data """
        return self._side_properties["dx"]
    @property
    def xcut(self):
        """ split value along the x-axes between two groups """
        return self._properties["xcut"]
    



