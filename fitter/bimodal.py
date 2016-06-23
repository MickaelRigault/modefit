#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

import numpy        as np
from scipy          import stats
import matplotlib.pyplot as mpl
# - local dependencies
from astrobject.utils.tools import kwargs_update
from .baseobjects import BaseModel,BaseFitter


__all__ = ["stepfit"]

# ========================= #
#  Main Methods             #
# ========================= #
def stepfit(x,data,errors,proba=None,dx=None,xcut=None,
                masknan=True,**kwargs):
    """ Load a StepFit instance to fit a step in your data """
    if masknan:
        flagnan = (x!=x) | (data !=data) | (errors != errors)
        return StepFit(x[~flagnan],data[~flagnan],errors[~flagnan],
                       proba=proba[~flagnan] if proba is not None else None,
                       dx=dx[~flagnan] if dx is not None else None,
                       xcut=xcut,**kwargs)
    
    return StepFit(x,data,errors,
                   proba=proba,dx=dx,xcut=xcut,
                   **kwargs)


# ========================== #
#                            #
#     Fitter                 #
#                            #
# ========================== #
class BimodalFit( BaseFitter ):
    """ """

    PROPERTIES         = ["data","errors","proba"]
    SIDE_PROPERTIES    = ["names"]
    DERIVED_PROPERTIES = []
    
    # ========================= #
    # = Initialization        = #  
    # ========================= #
    def __init__(self,data, errors, proba,
                 names=None, use_minuit=True,
                 modelname="Binormal"):
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
                                   
        proba: [array]
            Probability of the data to belong to one mode or the other.
            The probabilies must be float between 0 and 1
    
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
        self.set_data(data,errors,proba,names)
            
        # -- for the fit
        # use_minuit has a setter
        self.use_minuit = use_minuit
        self.set_model(eval("Model%s()"%modelname))

    def set_data(self,data,errors,proba,names=None):
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
        if (np.asarray(proba)>1).any() or (np.asarray(proba)<0).any():
            raise ValueErrors("probabilities (proba) must be between 0 and 1")
        
        if len(proba)!=len(data) or len(errors)!= len(data):
            raise ValueErrors("data, errors and proba must have the same size")
        
        # -- Warning -- #
        if names is not None and len(names) != len(data):
            warnings.warn("names size does not match the data one. => names ignored")
            names = None

        self._properties["data"] = np.asarray(data)
        self._properties["errors"] = np.asarray(errors)
        self._properties["proba"] = np.asarray(proba)

        self._side_properties["names"] = np.asarray(names) if names is not None else None
        
        
    # ========================= #
    # = Fit                   = #  
    # ========================= #

    def fit(self,use_minuit=None,**kwargs):
        """ fit the data using the bimode function. Inherite virtualfitter. """
        
        super(BimodalFit,self).fit(use_minuit=use_minuit,**kwargs)
        # ---------------------- #
        # -- Convinient result - #
        # ---------------------- #
        self.modelStep    = self.fitout["a"]['mean'] - self.fitout["b"]['mean']
        # -- Covariance needs to be added
        self.modelStepErr = np.sqrt(self.fitout["a"]['mean.err']**2 +
                                   self.fitout["b"]['mean.err']**2)

    def _fit_readout_(self):
        """
        """
        super(BimodalFit,self)._fit_readout_()
        self.fitout = {"a":{
            "mean":self.fitvalues["mean_a"],
            "mean.err":self.fitvalues["mean_a.err"],
            "sigma":self.fitvalues["sigma_a"],
            "sigma.err":self.fitvalues["sigma_a.err"]
            },
            "b":{
            "mean":self.fitvalues["mean_b"],
            "mean.err":self.fitvalues["mean_b.err"],
            "sigma":self.fitvalues["sigma_b"],
            "sigma.err":self.fitvalues["sigma_b.err"]
            }}
        

    def get_modelchi2(self,parameters):
        """ get the associated -2 log Likelihood

        This should usually be passed to the model with loading it.
        (See set_model)
        
        Parameters
        ----------
        
        parameters: [array]
            a list of parameter as they could be understood
            by self.model.setup to setup the current model.
                                   
        Return
        -------
        float (-2*log(likelihood))
        """
        self.model.setup(parameters)
        return -2 * self.model.get_loglikelihood(self.data,self.errors,self.proba)

    # ====================== #
    # Properties             #
    # ====================== #
    @property
    def data(self):
        """ Data used for the fit """
        return self._properties["data"]
    
    @property
    def errors(self):
        """ Errors associated to the data """
        return self._properties["errors"]

    @property
    def proba(self):
        """ probability to belog to the first group """
        return self._properties["proba"]
    
    @property
    def proba_r(self):
        """ probability to belog to the second group (1-propa)"""
        return 1-self.proba
    
    @property
    def names(self):
        """ Names associated to the data - if set """
        return self._side_properties["names"]
    
    # ---------------
    # - On the flight
    @property
    def npoints(self):
        return len(self.data)
        
# ========================== #
#                            #
#     The Model              #
#                            #
# ========================== #
class ModelBinormal( BaseModel ):
    """ Model for a Bi Normal Distribution (2 Gaussians) """
    
    FREEPARAMETERS = ["mean_a","sigma_a",
                      "mean_b","sigma_b"]
    # -------------------
    # - Initial Guesses    
    sigma_a_guess = 0
    sigma_a_fixed = False
    
    sigma_b_guess = 0
    sigma_b_fixed = False

    def setup(self,parameters):
        """ """
        self.mean_a,self.sigma_a,self.mean_b,self.sigma_b = parameters
        
    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx,p):
        """ Measure the likelihood to find the data given the model's parameters """
        Li = p * stats.norm.pdf(x,loc=self.mean_a,scale=np.sqrt(self.sigma_a**2 + dx**2)) + \
               (1-p) * stats.norm.pdf(x,loc=self.mean_b,scale=np.sqrt(self.sigma_b**2 + dx**2))
        
        return np.sum(np.log(Li))
    
    def get_case_likelihood(self,xi,dxi,pi):
        """ return the log likelihood of the given case. See get_loglikelihood """
        return self.get_loglikelihood([xi],[dxi],[pi])

    # ----------------------- #
    # - Bayesian methods    - #
    # ----------------------- #
    def lnprior(self,parameter):
        """ so far a flat prior """
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
                 use_minuit=None,**kwargs):
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
        self._properties["x"]  = np.asarray(x)
        self._side_properties["x"] = dx if dx is None else np.asarray(dx)
        self._properties["xcut"] = xcut
        # -- The Probability given the step location
        if proba is None:
            if xcut is None:
                raise ValueErrors("You need to give either proba or xcut to enable to define proba")
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
    def bokeh(self,savefile=None,name=None):
        """
        """
        from bokeh.models import ColumnDataSource, OpenURL, TapTool
        from bokeh.plotting import figure, output_file, show
    
        output_file("openurl.html")

        p = figure(plot_width=400, plot_height=400,
                   toolbar_location="above",
                   tools="tap,resize,pan", title="Click the Dots")
        
        source = ColumnDataSource(data=dict(
            x=self.x,
            y=self.data,
            name=self.name,
            #color=["navy", "orange", "olive", "firebrick", "gold"]
            ))

        p.circle('x', 'y',  size=20, source=source)

        url = "http://snf.in2p3.fr/lc/@snname"
        taptool = p.select(type=TapTool)
        taptool.callback = OpenURL(url=url)

        show(p)

    def show(self,savefile=None,axes=None,#rangey=[-0.6,0.6],
             figure=None,cmap=mpl.cm.ocean,ybihist=True,
             propaxes={}, rangex=None,rangey=None,
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
            from astrobject.utils.mpladdon import add_threeaxes
            
            fig = figure if figure is not None else mpl.figure(figsize=[7,5])
            ax,axhistx,axhisty = fig.add_threeaxes(**propaxes)

        # =================
        # The Scatter Plot
        # =================
        ecolor = kwargs.pop("ecolor","0.7")
        ax.errorbar(self.x,self.data,xerr=self.dx,yerr=self.errors,
                    ecolor=ecolor,ls="None", marker=None,label="_no_legend_", zorder=2)

        prop = kwargs_update({"s":80,"edgecolors":"0.7","linewidths":1,"zorder":5,},
                             **kwargs)
        ax.scatter(self.x,self.data,c=self.proba,cmap=cmap,
                   **prop)
        if self.xcut is not None:
            ax.axvline(self.xcut,ls="--",color="k",alpha=0.8)

        # =================
        # The Histograms
        # =================
        propa = {"histtype":"step","fc":cmap(0.9,0.4),"ec":"k","fill":True}
        propb = {"histtype":"step","fc":cmap(0.1,0.4),"ec":"k","fill":True}
        
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
                axhisty.set_xlim(-axhisty.get_xlim()[-1],axhisty.get_xlim()[-1])
                
        # =================
        # The Fitted Values
        # =================
        #-- if you already made the fit
        if self.has_fit_run():
            # -- To be improve, this does not move with the axis if user does so.
            from astrobject.utils.mpladdon import hline,hspan
            for ax_,cut in [[ax,self.xcut],[axhisty,0]]:
                if ax_ is None: continue
                
                ax_.hline(self.fitout["a"]['mean'],xmax=cut,
                       color=mpl.cm.binary(0.9),alpha=0.8)
            
                ax_.hspan(self.fitout["a"]['mean']-self.fitout["a"]['mean.err'],
                        self.fitout["a"]['mean']+self.fitout["a"]['mean.err'],
                        xmax=cut,
                        color=mpl.cm.binary(0.9),alpha=0.2)
            
                ax_.hline(self.fitout["b"]['mean'],xmin=cut,
                     color=cmap(0.1,0.8))
                
                ax_.hspan(self.fitout["b"]['mean']-self.fitout["b"]['mean.err'],
                        self.fitout["b"]['mean']+self.fitout["b"]['mean.err'],
                        xmin=cut,
                        color=cmap(0.1,0.2))
        # ---------
        # - Plots
        self._plot = {}
        self._plot["ax"] = [ax,axhistx,axhisty]
        self._plot["fig"] = fig
        return self._plot
    # ========================= #
    # = Properties            = #  
    # ========================= #
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
    
