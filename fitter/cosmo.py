#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

import numpy        as np
from scipy          import stats
import matplotlib.pyplot as mpl
# - astropy
from astropy import cosmology
# - local dependencies
from astrobject.utils.tools import kwargs_update
from .baseobjects import BaseModel,BaseFitter, DataHandler


PLANCK_H0 = 67.7
PLANCK_Om = 0.307

__all__ =["Delta_Cosmo"]



class Delta_Cosmo( BaseFitter, DataHandler ):
    """ """
    PROPERTIES = ["redshift"]
    
    def __init__(self,redshift, data, errors,
                 names=None, use_minuit=True,
                 modelname="w0wa"):
        """  low-level class to enable to fit a unimodal model on data
        the given their errors.

        Parameters
        ----------
        redshift, data, errors: [array,array,array]
            redshift, magnitude and magnitude errors
                                   
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
        self.set_data(redshift, data,errors,names)
            
        # -- for the fit
        # use_minuit has a setter
        self.use_minuit = use_minuit
        self.set_model(eval("Model%s()"%modelname))


    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self,redshift, data, errors, names=None):
        """ set the information for the fit.

        Parameters
        ----------
        redshift, data, errors: [array,array,array]
            redshift, magnitude and magnitude errors
            
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

        self._properties["redshift"] = np.asarray(redshift)
        self._properties["data"] = np.asarray(data)
        self._properties["errors"] = np.asarray(errors)
        self._side_properties["names"] = np.asarray(names) if names is not None else None

    def _get_model_args_(self):
        """ """
        return self.redshift,self.data,self.errors

    # --------------- #
    #   PLOTTER
    # --------------- #
    def show(self,savefile=None,show=True):
        """ """
        from astrobject.utils.mpladdon import figout
        import matplotlib.pyplot as mpl

        fig = mpl.figure(figsize=[10,8])
        ax  = fig.add_subplot(111)
        ax.plot(self.redshift, self.data,"bo")
        ax.errorbar(self.redshift, self.data, yerr=self.errors,
                    ls="None", ecolor="0.7")
        
        fig.figout(savefile=savefile, show=show)


    # ------------------ #
    #   Plot Add on      #
    # ------------------ #
    def display_mcmc_models(self,ax, nsample, z, color=None,
                            basecolor=None, alphasample=0.07,**kwargs):
        """ add mcmc model reconstruction to the plot

        Parameters:
        -----------
        ax: [matplotlib.Axes]
            Where the model should be drawn

        nsample: [int]
            Number of test case for the mcmc walkers

        color: [matplotlib's color]
            Color of the main line drawing the 50% samples

        basecolor: [matplotlib's color]
            Color of the individual sampler (nsample will be displayed)

        **kwargs goes to matplotlib.pyplot
        Return
        ------
        list of matplotlib.plot return
        """
        import matplotlib.pyplot as mpl
        if not self.mcmc.has_chain():
            raise AttributeError("You need to run MCMC first.")
        
        if color is None:
            color     = mpl.cm.Blues(0.8,1)
        if basecolor is None:
            basecolor = mpl.cm.Blues(0.5,0.1)
        
        pl = [self.display_model(ax, param, z, color=basecolor,**kwargs)
              for param in self.mcmc.samples[np.random.randint(len(self.mcmc.samples), size=100)] ]
        
        self.display_model(ax, np.asarray(self.mcmc.derived_values).T[0], z,
                           color=color, **kwargs)
        
        return pl
    
    def display_model(self,ax,parameters, z,
                      color="r", ls="-",**kwargs):
        """ add the model on the plot
        Parameters
        ----------
        ax: [matplotlib.Axes]
            where the model should be drawn
            
        parameters: [array]
            parameters of the model (similar to the one fitted)
            
        Return
        ------
        """
        self.model.setup(parameters)
        ymodel = self.get_model(z)
        #ymodel[~self.model.lbdamask] = np.NaN
        return ax.plot(z, ymodel, color=color,ls=ls,
                       **kwargs)

    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def redshift(self):
        """ """
        return self._properties["redshift"]


    
class Modelw0wa( BaseModel ):
    """ """
    PROPERTIES = ["refcosmo","w0","wa"]
    FREEPARAMETERS = ["w0","wa"]
    # -------- #
    #  SETTER  #
    # -------- #
        
    def setup(self, parameters):
        """ """
        self._properties["w0"],self._properties["wa"] = parameters

    def get_model(self, z):
        """ """
        return self.cosmo.distmod(z).value - self.refcosmo.distmod(z).value - 0.08
    
    
    def get_loglikelihood(self, z, y, dy):
        """ Measure the likelihood to find the data given the model's parameters.
        Set pdf to True to have the array prior sum of the logs (array not in log=pdf) """
        
        res = y - self.get_model(z)
        return -0.5 * np.sum(res**2/dy**2) 

    # ---------- #
    # - Priors - #
    # ---------- #
    def lnprior(self,parameters):
        """ """
        return 0

    @property
    def _mcmc_initbounds(self):
        return [[-1.5,1.5],[-0.5,0.5]]
        
    # =============== #
    #  Properties     #
    # =============== #

    @property
    def w0(self):
        return self._properties["w0"]

    @property
    def wa(self):
        return self._properties["wa"]
    
    @property
    def cosmo(self):
        return cosmology.Flatw0waCDM(PLANCK_H0,PLANCK_Om, w0=self.w0, wa=self.wa)
    
    @property
    def refcosmo(self):
        """ """
        if self._properties["refcosmo"] is None:
            self._properties["refcosmo"] = cosmology.FlatLambdaCDM(PLANCK_H0,PLANCK_Om)
            
        return self._properties["refcosmo"]
    
    
