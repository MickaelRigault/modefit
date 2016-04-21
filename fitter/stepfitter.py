#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as N
from scipy       import stats
from ..lowlevel.bimodalfitter import BimodalFit

__all__ = ["step_fitter"]


def step_fitter(x,data,errors,proba=None,dx=None,xCut=None,**kwargs):
    """ Load a StepFit instance to fit a step in your data """
    return StepFit(x,data,errors,proba=None,dx=None,xCut=None,**kwargs)

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
                 xCut=None,**kwargs):
        """
        = This class is a child of *BimodalFit* that have the ability to
          use an extra x-axis value to potentially define *proba* if needed
          and to show a *x,data* plot taking into account this probability.
          See further details in ProbaFit_library.BimodalFit
        =
        x: [array]                 The x-axis where the cut is made between the
                                   2 populations. If *proba* is not pre-defined,
                                   this will be used together with xCut to do it.
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
                                   If None, *x* and *xCut* (with *dx* if any)
                                   will be used to defined *proba*.
                                   (See self.get_proba)
                                   If not None, *proba* must have the same size
                                   as *data*. The probabilies must be float between
                                   0 and 1.
                                   
        dx: [array/None]           If the x-axis have errors.
                                   In addition to plot functions, these errors
                                   will be used if proba have to be defined using
                                   *x* and *xCut*. CAUTION, this will assume symetric
                                   gaussian errors. (See self.get_proba).
                                   This must have the same size as *data (x)* 
        
        xCut: [float]              This is were the split is between the 2 populations.
                                   In addition to plot functions, this value will be
                                   used if proba have to be defined based on this and
                                   the corresponding *x* values (and *dx* if
                                   any self.get_proba).
                                   IMPORANT: if *proba* is not given. *xCut* must be.


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
        self.xCut = xCut
        # -- The Probability given the step location
        if proba is None:
            if xCut is None:
                raise ValueError("You need to give either proba or xCut to enable to define proba")
            proba = self.get_proba()

        # ----------------------- #
        # -- The Mother's Init -- #
        # ----------------------- #
        super(StepFit,self).__init__(data, errors,proba,**kwargs)
        

    # ========================= #
    # = Step Stuffs           = #  
    # ========================= #
    def get_proba(self,xCut=None):
        """
        =
         This function will split the sample in two at the given *xCut* value.
         If no *dx* provided, *proba* will be 0 or 1. Otherwise errors
         on the x-axis will be used to define non-trivial probability.
         CAUTION: if there is errors, this assumes then symetric and gaussian.
        =

        xCut: [float/None]         The x-value where the sample is splitted in 2.
                                   If None is set, self.xCut will be used if it
                                   already is defined. This will update self.xCut.

        = RETURNS =
        array of float (between 0 and 1 ;size of x)
        """
        if xCut is not None:
            self.xCut = xCut
        
        if self.dx is None: # - faster this way
            return N.asarray([0 if x>self.xCut else 1 for x in self.x])
        
        return N.asarray([stats.norm(loc=x,scale=dx).cdf(self.xCut)
                         for x,dx in zip(self.x,self.dx)])

    # ========================= #
    # = Step Shows            = #  
    # ========================= #
    def show_proba(self,*args,**kwarks):
        """
        = This is to do the *proba*, *data* 3-axes plot, which corresponds
          to the mother-class ProbaFit_library.BimodalFit.show
        =

        *args and **kwargs          goes to the mother's class function show()
                                    (e.g., savefile, axes, ...)
                                    
        """
        super(StepFit,self).show(*args,**kwarks)

        
    def show(self,savefile=None,axes=None,rangey=[-0.6,0.6],
             **kwargs):
        """
        = Plot the *x*, *data* 3-axes plot, including *proba* marker
          colors.
        =

        savefile: [string/None]     If you wish to save the plot in the given
                                    filename (*without extention*). This will create
                                    a `savefile`.pdf and `savefile`.png
                                    The plot won't be show if it is saved.

        axes: [3-array axes/None]   Give mpl axes in which the plot will be drawn
                                    3 axes must be given (axsc, axhistx, axhisty).
                                    The latest 2 (both or only one) could be set to
                                    None if you do not wish to have the corresponding
                                    histogram plotted.

        # ------ #
        
        **kwargs                    goes to the StepPlot Class function show_key
                                    (e.g., swap_bihistograms, catch_names or any
                                    matplotlib.plot entry ...)

        = RETURNS =
        Void
        """
        self._load_plot_()
        self.plot.show_key("x",dkey="dx",axes=axes,
                           rangey=rangey,
                           #legendprob=dict(label="Proba"),
                           **kwargs)
        if self.xCut is not None:
            self.plot.axvline(self.plot.ax,self.xCut,ls="--",color="k",alpha=0.8)
        
        #-- if you already made the fit
        if "fitModelA" in dir(self):
            # -- To be improve, this does not move with the axis if user does so.
            self.plot.axhline(self.plot.ax,self.fitModelA['mean'],xmax=self.xCut,
                                color=pb.P.cm.binary(0.9),alpha=0.8)
            
            self.plot.axhspan(self.plot.ax,
                              self.fitModelA['mean']-self.fitModelA['mean.err'],
                              self.fitModelA['mean']+self.fitModelA['mean.err'],
                              xmax=self.xCut,
                              color=pb.P.cm.binary(0.9),alpha=0.2)
            
            self.plot.axhline(self.plot.ax,self.fitModelB['mean'],xmin=self.xCut,
                              color=pb.P.cm.Blues(0.9),alpha=0.8)
            self.plot.axhspan(self.plot.ax,
                              self.fitModelB['mean']-self.fitModelB['mean.err'],
                              self.fitModelB['mean']+self.fitModelB['mean.err'],
                              xmin=self.xCut,
                              color=pb.P.cm.Blues(0.9),alpha=0.2)
            
        self.plot.savefilereader(savefile)
