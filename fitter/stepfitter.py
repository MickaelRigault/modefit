#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as N
from scipy       import stats
import matplotlib.pyplot as mpl
from ..lowlevel.bimodalfitter import BimodalFit

__all__ = ["step_fitter"]


def step_fitter(x,data,errors,proba=None,dx=None,xcut=None,
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
# ========================== #
# ==  Step                == #
# ========================== #
# ========================== #
class StepFit( BimodalFit ):
    """
    """
    def __init__(self,x,data,errors,
                 proba=None,dx=None,
                 xcut=None,**kwargs):
        """
        = This class is a child of *BimodalFit* that have the ability to
          use an extra x-axis value to potentially define *proba* if needed
          and to show a *x,data* plot taking into account this probability.
          See further details in ProbaFit_library.BimodalFit
        =
        x: [array]                 The x-axis where the cut is made between the
                                   2 populations. If *proba* is not pre-defined,
                                   this will be used together with xcut to do it.
                                   This must have the same size as *data*

        data: [array]              The data that potentially have a bimodal
                                   distribution (like a step).
                                   In case of Cosmological fit, this could be
                                   the Hubble Residual for instance.
                                   
        errors: [array]            Errors associated to the *data*.
                                   This must have the same size as *data*
                                   
        # ------- #
                              
        proba: [array/None]        Probability of the data to belong to one
                                   mode or the other.
                                   If None, *x* and *xcut* (with *dx* if any)
                                   will be used to defined *proba*.
                                   (See self.get_proba)
                                   If not None, *proba* must have the same size
                                   as *data*. The probabilies must be float between
                                   0 and 1.
                                   
        dx: [array/None]           If the x-axis have errors.
                                   In addition to plot functions, these errors
                                   will be used if proba have to be defined using
                                   *x* and *xcut*. CAUTION, this will assume symetric
                                   gaussian errors. (See self.get_proba).
                                   This must have the same size as *data (x)* 
        
        xcut: [float]              This is were the split is between the 2 populations.
                                   In addition to plot functions, this value will be
                                   used if proba have to be defined based on this and
                                   the corresponding *x* values (and *dx* if
                                   any self.get_proba).
                                   IMPORANT: if *proba* is not given. *xcut* must be.


        **kwargs                   goes to the mother __init__ function
                                   (ProbaFit_library.BimodalFit ; e.g., use_minuit,
                                   modelName ...)

        = RETURNS =
        Void, define the object
        """
        # -- Init Tests -- #
        if len(x) != len(data):
            raise ValueError("x and data must have the sample size (%d vs. %d)"%(len(x),len(data)))
        
        if dx is not None and len(x) != len(dx):
            raise ValueError("x and dx must have the sample size (%d vs. %d). Give dx=None not to use it"%(len(x),len(dx)))

        # ----------------------- #
        # -- The x Information -- #
        # ----------------------- #
        # -- basic x-stuffs
        self.x  = N.asarray(x)
        self.dx = dx if dx is None else N.asarray(dx)
        self.xcut = xcut
        # -- The Probability given the step location
        if proba is None:
            if xcut is None:
                raise ValueError("You need to give either proba or xcut to enable to define proba")
            proba = self.get_proba()

        # ----------------------- #
        # -- The Mother's Init -- #
        # ----------------------- #
        super(StepFit,self).__init__(data, errors, proba, **kwargs)
        

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
            self.xcut = xcut
        
        if self.dx is None: # - faster this way
            return N.asarray([0 if x>self.xcut else 1 for x in self.x])
        
        return N.asarray([stats.norm(loc=x,scale=dx).cdf(self.xcut)
                         for x,dx in zip(self.x,self.dx)])
    
    # ========================= #
    # = Step Shows            = #  
    # ========================= #
    
    def show(self,savefile=None,axes=None,rangey=[-0.6,0.6],
             figure=None,cmap=mpl.cm.ocean,ybihist=True,
             propaxes={},**kwargs):
        """
        Plot the *x*, *data* 3-axes plot, including *proba* marker colors.

        Parameters:
        -----------

        savefile: [string/None]     If you wish to save the plot in the given
                                    filename (*without extention*). This will create
                                    a `savefile`.pdf and `savefile`.png
                                    The plot won't be show if it is saved.

        axes: [3-array axes/None]   Give mpl axes in which the plot will be drawn
                                    3 axes must be given (axsc, axhistx, axhisty).
                                    The latest 2 (both or only one) could be set to
                                    None if you do not wish to have the corresponding
                                    histogram plotted.
                                    
        figure: [mpl.Figure]        if you did not provided axes, you can give a figure
                                    into which the axes will be drawn, otherwise this
                                    this create a new one.
        # ------ #

        ybihist: [bool]             Oppose the two y-histogram
        
        propaxes: [dict]            properties entering the 'add_threeaxes' as kwargs.
        
        **kwargs                    goes to the StepPlot Class function show_key
                                    (e.g., swap_bihistograms, catch_names or any
                                    matplotlib.plot entry ...)

        Returns:
        --------
        Void
        """
        from ..utils.tools import kwargs_update

        # =================
        # Setting the Axes
        # =================
        if axes is not None:
            if len(axes) != 3:
                raise ValueError("the input 'axes' must be a 3d-array (ax,axhistx,axhisty)")
            ax,axhistx,axhisty = axes
            fig = ax.figure
        else:
            from ..utils.mpladdon import add_threeaxes
            fig = figure if figure is not None else mpl.figure(figsize=[13,10])
            ax,axhistx,axhisty = fig.add_threeaxes(**propaxes)

        # =================
        # The Scatter Plot
        # =================
        ecolor = kwargs.pop("ecolor","0.7")
        ax.errorbar(self.x,self.data,xerr=self.dx,yerr=self.error,
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
        if axhistx is not None:
            axhistx.hist(self.x,weights=1-self.proba,**propb)
            axhistx.hist(self.x,weights=self.proba,**propa)
            
        if axhisty is not None:
            axhisty.hist(self.data,weights=1-self.proba,orientation="horizontal",
                         **propb)
            axhisty.hist(self.data,weights=self.proba*(-1 if ybihist else 1),
                         orientation="horizontal",
                         **propa)
            
            if ybihist:
                axhisty.set_xlim(-axhisty.get_xlim()[-1],axhisty.get_xlim()[-1])
                
        # =================
        # The Fitted Values
        # =================
        #-- if you already made the fit
        if self.fitperformed:
            # -- To be improve, this does not move with the axis if user does so.
            from ..utils.mpladdon import hline,hspan
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
