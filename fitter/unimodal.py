#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module based on virtualfitter to fit bimodal distributions. Does not have to be a step """

import numpy        as np
from scipy          import stats
import matplotlib.pyplot as mpl
# - local dependencies
from ..utils.tools  import kwargs_update
from .baseobjects import BaseModel,BaseFitter

__all__ = ["normalfit"]

# ========================= #
#  Main Methods             #
# ========================= #
def normalfit(data,errors,names=None,
            masknan=True,**kwargs):
    """ Adjuste the weighted mean and intrinsic dispersion
    on the data"""
    if masknan:
        flagnan = (data!=data) | (errors !=errors)
        return UnimodalFit(data[~flagnan],errors[~flagnan],
                           names=names[~flagnan] if names is not None else None,**kwargs)
    
    return UnimodalFit(data,errors,names=names,**kwargs)

# ========================== #
#                            #
#     Fitter                 #
#                            #
# ========================== #
class UnimodalFit( BaseFitter ):
    """ """
    PROPERTIES         = ["data","errors"]
    SIDE_PROPERTIES    = ["names"]
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
        

    # =================== #
    #   Main              #
    # =================== #
    def get_modelchi2(self,parameters):
        """ Parses the parameters and return the associated -2 log Likelihood
        Both the parser and the log Likelohood functions belongs to the
        model.
        This should usually be passed to the model with loading it. 
        
        parameters: [array]
            a list of parameter as they could be understood
            by self.model.setup to setup the current model.
                                   
        Returns
        -------
        float (chi2 defines as -2*log_likelihood)
        """
        self.model.setup(parameters)
        return -2 * self.model.get_loglikelihood(self.data,self.errors)
    
    # =================== #
    #  Properties         #
    # =================== #
    @property
    def data(self):
        return self._properties["data"]
    
    @property
    def errors(self):
        return self._properties["errors"]

    @property
    def names(self):
        return self._side_properties["errors"]

    @property
    def npoints(self):
        return len(self.data)



class ModelNormal( BaseModel ):
    """
    """
    FREEPARAMETERS = ["mean","sigma"]
        
    def setup(self,parameters):
        """ """
        self.mean,self.sigma = parameters
        
    # ----------------------- #
    # - LikeLiHood and Chi2 - #
    # ----------------------- #
    def get_loglikelihood(self,x,dx):
        """
        """
        return np.sum([ np.log(self._get_case_likelihood_(x_,y_))
                       for x_,y_ in zip(x,dx)] )
    
    def _get_case_likelihood_(self,x,dx):
        """
        """
        return stats.norm(loc=self.mean,scale=np.sqrt(self.sigma**2 + dx**2)).pdf(x)


    # ----------------------- #
    # - Bayesian methods    - #
    # ----------------------- #
    def lnprior(self,parameter):
        """ so far a flat prior """
        return 0

    # ----------------------- #
    # - Model Particularity - #
    # ----------------------- #
    def _minuit_chi2_(self,mean,sigma):
        """
        """
        parameter = mean,sigma
        return self.get_chi2(parameter)
