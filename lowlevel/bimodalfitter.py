#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

import numpy        as np
from scipy          import stats

from ..utils.tools  import kwargs_update
from .virtualfitter import ScipyMinuitFitter,VirtualFitter

"""
import plotBasics as pb
from mplTools     import get_mrearth,FancyLine
"""


#############################
#  GENERAL TOOLS            #
#############################

# ========================== #
# ========================== #
# ==  Fitter              == #
# ========================== #
# ========================== #
class BimodalFit( VirtualFitter ):
    """
    =  Child of VirtualFitter that understand the scipy/minuit tricks
       of the Models that inherit ScipyMinuitFitter.
    = 
    """
    def __init__(self,data, errors,
                 proba,names=None,
                 use_minuit=True,
                 modelName="Binormal"):
        """
        = This class is a low-level class that enable to fit a *bimodel model*
          to the given *data* (*errors*) that have probality (*proba*) to
          belong to one mode or an other. The models are defined in seperated
          classes (e.g. ModelBinormal)
        =

        data: [array]              The data that potentially have a bimodal
                                   distribution (like a step).
                                   In case of Cosmological fit, this could be
                                   the Hubble Residual for instance.
                                   
        errors: [array]            Errors associated to the *data*.
                                   This must have the same size as *data*
                                   
        proba: [array]             Probability of the data to belong to one
                                   mode or the other.
                                   This must have the same size as *data*
                                   The probabilies must be float between 0 and 1

        # ------- #
    
        names: [string-array/None] Names associated with the data.
                                   This enable to follow the data more easily.
                                   If this is given, you shall be able to use
                                   interactive ploting and see which points
                                   corresponds to which object.

        use_minuit: [bool]         Set the technique used to fit the *model* to
                                   the *data*. Minuit is the iminuit library. If not
                                   used, the scipy minimisation is used.

        modelName: [string]        The name of the class used to define the bimodal
                                   *model*. This name must be an existing class of this
                                   library.

        = RETURNS =
        Void, define the object
        """
        # ------------------------ #
        # -- Fatal Input Errors -- #
        if (np.asarray(proba)>1).any() or (np.asarray(proba)<0).any():
            raise ValueError("Proba must be between 0 and 1")
        if len(proba)!=len(data) or len(errors)!= len(data):
            raise ValueError("Data, Error and Proba must have the same size")
        
        # -- Warning -- #
        if names is not None and len(names) != len(data):
            print "WARNING, `names` size doesnot match `data` one. `names` set to None"
            names = None
            
        # ------------------------ #
        # --- Setup the object --- #
        # -- Basics
        self.data    = np.asarray(data)
        self.error   = np.asarray(errors)
        self.proba   = np.asarray(proba)
        if names is not None:
            self.names   = np.asarray(names)
        else:
            self.names   = None
            
        # -- Derived
        self.aproba  = 1. - self.proba
        self.var     = self.error **2
        self.npoints = len(self.data)

        # -- for the fit
        self.use_minuit = use_minuit
        self.load_model(modelName=modelName)


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
        
    # ========================= #
    # = Model                 = #  
    # ========================= #
    def load_model(self,modelName,**kwargs):
        """
        = This loads the *model* object base on the given name =

        modelName: [string]        The name of the class used to define the bimodal
                                   *model*. This name must be an existing class of this
                                   library (or an imported one)

        **kwargs                   goes to self.set_guesses.
                                   (Define the guess, fixed and boundaries values)
        = RETURNS =
        Void, defines *model* and associated objects
        """
        self.model      = eval("Model%s()"%modelName)
        self._modelName = modelName
        self.model.get_chi2 = self.get_modelChi2
        self.setup_guesses(**kwargs)

        
    def get_modelChi2(self,parameters):
        """
        = Parses the parameters and return the associated -2 log Likelihood
          Both the parser and the log Likelohood functions belongs to the
          model.
          This should usually be passed to the model with loading it.
          (See load_model)
        =
        
        parameters: [array]        a list of parameter as they could be understood
                                   by self.model.setup to setup the current model.
                                   
        = RETURNS =
        float (-2*log(likelihood) for *data*, *errors* and *proba* and the given model)
        
        """
        self.model.setup(parameters)
        return -2 * self.model.get_loglikelihood(self.data,self.error,self.proba)

    
    # ========================= #
    # = Ploting Tools         = #
    # ========================= #
    def _load_plot_(self):
        """
        = Internal function loading a *PlotStep* object in self.plot =
        """
        raise NotImplementedError("To Be Done")
        if "plot" not in dir(self):
            self.plot = PlotStep(self)
    
    
    def show(self,savefile=None,
             catch_names=True,axes=None,
             **kwargs):
        """
        = Visuallisation of the *data* and the models.
          This function makes use of the PlotStep class.
        =

        savefile: [string/None]     If you wish to save the plot in the given
                                    filename (*without extention*). This will create
                                    a `savefile`.pdf and `savefile`.png
                                    The plot won't be show if it is saved.

        catch_names: [bool]         If the object have names, you will be able to pick
                                    on the plot the marker to know which corresponds
                                    to what.
                                    This does not work if the plot is saved instead
                                    of shown or if self.names is not defined.

        axes: [3-array axes/None]   Give matplotlib axes in which the plot will be
                                    drawn 3 axes must be given (axsc, axhistx, axhisty)
                                    The latest 2 (both or only one) could be set to
                                    None if you do not wish to have the corresponding
                                    histogram plotted.

        **kwargs                    goes to PlotStep.show_key (via self.plot)
        
        """
        self._load_plot_()
        self.plot.show_key("proba",axes=axes,**kwargs)
        
        # -- Mean best fit results
        if "fitout" in dir(self):
            self.plot.fline = pt.FancyLine()
            
            self.plot.fline.hline(self.plot.ax,self.fitout["a"]['mean'],
                                rangex=[0,1],lws=[0,1],alphas=[0,1],colors=[0,1],
                                cmap=pt.P.cm.binary)
            
            self.plot.fline.hline(self.plot.ax,self.fitout["b"]['mean'],
                                rangex=[0,1],lws=[1,0],alphas=[1,0],colors=[1,0],
                                cmap=pt.P.cm.Blues)
            
        # -- the output
        self.plot.savefilereader(savefile)



    # ====================== #
    # Properties             #
    # ====================== #

        
# ========================== #
# ========================== #
# ==  The Model           == #
# ========================== #
# ========================== #
class ModelBinormal( ScipyMinuitFitter ):
    """
    """
    freeParameters = ["mean_a","sigma_a",
                      "mean_b","sigma_b"]
        
    sigma_a_guess = 0
    sigma_a_fixed = False
    # sigma_a_limit = [None,None]
    
    sigma_b_guess = 0
    sigma_b_fixed = False

    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx,p):
        """
        """
        return np.sum([ np.log(self._get_case_likelihood_(x_,y_,p_))
                       for x_,y_,p_ in zip(x,dx,p)] )
    
    def _get_case_likelihood_(self,x,dx,p):
        """
        """
        return p * self.get_modela_pdf(x,dx) + (1-p) * self.get_modelb_pdf(x,dx)

    # ----------------------- #
    # - The ActualModel     - #
    # ----------------------- #
    def get_modela_pdf(self,x,dx):
        """
        """
        return stats.norm(loc=self.mean_a,scale=np.sqrt(self.sigma_a**2 + dx**2)).pdf(x)

    def get_modelb_pdf(self,x,dx):
        """
        """
        return stats.norm(loc=self.mean_b,scale=np.sqrt(self.sigma_b**2 + dx**2)).pdf(x)

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
# ========================== #
# ==  Plot Class          == #
# ========================== #
# ========================== #

class PlotData():

    def change_cmap(self,cmap):
        """
        """
        self.cmap=cmap
        self.default_scatter_kwargs  = kwargs_update(self.default_scatter_kwargs,
                                                    **{"cmap":self.cmap})
        self._histX_kwargs_A = kwargs_update(self.default_histx_kwargs,
                                **{"fc":self.cmap(1.),"alpha":0.8})
        self._histX_kwargs_B = kwargs_update(self.default_histx_kwargs,
                                **{"fc":self.cmap(0.),"alpha":0.8})
        
        self._histY_kwargs_A = kwargs_update(self.default_histy_kwargs,
                                **{"fc":self.cmap(1.),"alpha":0.8})
        self._histY_kwargs_B = kwargs_update(self.default_histy_kwargs,
                                **{"fc":self.cmap(0.),"alpha":0.8})
            
    def show_key(self,key,dkey=None,axes=None,catch_names=False,
                 swap_bihistograms=False,reset=True,
                 legend=True,legendprop={},
                 rangey=None,rangex=None,
                 ybihist = True,xbins=10,ybins=10,
                 **kwargs):
            
        ValueX    = eval("self.obj.%s"%key)
        ValueXerr = eval("self.obj.%s"%dkey) if dkey is not None else None
        # -- Scatter Plot
        k = kwargs_update(self.default_scatter_kwargs,**kwargs)
        self.reset()
        self.set_axes(with_histx=False,with_histy=False,
                      force_axhist=False)
        self.scatter(ValueX,self.obj.data,
                     dy=self.obj.error,
                     dx=ValueXerr,
                     add_colorbar=False,
                     scolor=self.obj.proba,
                     ax=self.ax,**k)

        self.set_axhistx()
        self.set_axhisty()
        # ----------------- #
        # - Histograms    - #
        # ----------------- #
        self._histX_kwargs_A["range"] =  rangex
        self._histX_kwargs_B["range"] =  rangex
        self._histY_kwargs_A["range"] =  rangey
        self._histY_kwargs_B["range"] =  rangey
        # -- X
        if self.axhistx is not None:
            b = self.hist(ValueX,weights=self.obj.proba,
                          ax=self.axhistx,bins=xbins,
                          
                          **self._histX_kwargs_A)
            
            b = self.hist(ValueX,weights=self.obj.aproba,
                          ax=self.axhistx,bins=xbins,
                          **self._histX_kwargs_B)
            
        # -- Y
        if self.axhisty is not None:
            by = self.hist(self.obj.data,weights=self.obj.proba,
                           invert_axes=swap_bihistograms,
                           ax = self.axhisty,
                           bins=ybins,
                           **self._histY_kwargs_A)
            if ybihist:
                for p in self._hist[-1]:
                    p._path.vertices[:,0] *= -1
                
                                        
            by = self.hist(self.obj.data,weights=self.obj.aproba,
                           invert_axes=swap_bihistograms,
                           bins=ybins,ax=self.axhisty,
                           **self._histY_kwargs_B)
            
            if ybihist:
                self.axhisty.set_xlim(-self.axhisty.get_xlim()[-1],self.axhisty.get_xlim()[-1])
                         
        
        # -- Debug stuff
        #[self.ax.axhline(b,color="0.7") for b in by]
        #[self.axhisty.axhline(b,color="0.7") for b in by]
        vrange = ValueX.max()-ValueX.min()
        #self.set_xlim(ValueX.min() - 0.1*vrange,
        #              ValueX.max() + 0.1*vrange)
        self.set_ylim(*self.ax.get_ylim())
        if legend:
            self.add_legend(**legendprop)
        
    # -------------------- #
    # - LEGEND TRICKS    - #
    # -------------------- #
    def add_legend(self,cbarwidth=0.02,nticks=4,
                   label="",update=True,
                   axcbar=None,**kwargs):
        # -- are you only 0 or 1 ?
        if (self.obj.proba %1 >0).any(): # No there is proba
            # -- Where to show the bar
            if axcbar is None:
                self.axlengend =self.get_caxLegend(cbarwidth=cbarwidth)
            else:
                self.axlengend = axcbar
                
            # --Show the bar
            self.legendbar = pb.mympl.add_colorbar(self.axlengend,
                              self.cmap,
                              orientation="horizontal",**kwargs)
            # -- Ticks stuff -- #
            self.legendbar.set_ticks([])
            [self.axlengend.text(1.0-x*0.95, -.3,r"$%d$%%"%((1-x)*100),
                              fontsize="medium",
                              va="top",ha="center",
                              transform=self.axlengend.transAxes)
                for x in np.linspace(0,1,nticks)]
            
            # -- Labelling -- #
            self.axlengend.text(0.5,1.3,label,
                             fontsize="large",
                             va="bottom",ha="center",
                             transform=self.axlengend.transAxes)
                                    
            
            if update:
                self.fig.canvas.draw()
                
        else:
           print "Legend for trivial cases Not ready yet"

           

    def set_xlim(self,xmin,xmax):

        self.ax.set_xlim(xmin,xmax)
        if self.axhistx is not None:
            self.axhistx.set_xlim(self.ax.get_xlim())

    def set_ylim(self,ymin,ymax):

        self.ax.set_ylim(ymin,ymax)
        if self.axhisty is not None:
            self.axhisty.set_ylim(self.ax.get_ylim())

