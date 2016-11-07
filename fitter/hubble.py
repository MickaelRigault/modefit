#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" module to perform Hubble Diagram fit based on Type Ia Supernovae """

import numpy as np

import warnings
from scipy          import stats, linalg
import matplotlib.pyplot as mpl
from astropy import constants

# - local dependencies
from astrobject.utils.tools import kwargs_update

from .baseobjects import BaseModel, BaseFitter, DataSourceHandler
import astropy
try:
    from astropy.cosmology import Planck15 as DEFAULT_COSMO
except:
    from astropy.cosmology import Planck13 as DEFAULT_COSMO
    warnings.warn("Your astropy is not up to data. This will use Planck13 cosmology instead of Planck15")
    

__all__ = ["get_hubblefit"]

    
CLIGHT_km_s  = constants.c.to("km/s").value
PECULIAR_VELOCITY = 300 # in km/s

###############################
#                             #
#   Main Tools                #
#                             #
###############################
def get_hubblefit(data, corr=["x1,c"]):
    """ get an object that allows you to do Hubble fit (hubblizer)

    Parameters
    ----------
    data: [dict]
       This dictionary should have this format:
       {name: {k: value1, k.err: err1,
               k2: value2, k2.err, err2,
               k3: value3,
               cov_kk2: cov_between_k_and_k2,
               cov_k2k: cov_between_k_and_k2,
               etc.
               }
        empty entries will be assumed to be 0

        Requested parameters:
         - mag
         - zcmb

        Any list of k, k1,k2 could be used for standardization except
        the requested parameters (mag, zcmb)

    corr: [list of string] -optional-
        List and k-values (see data) that you want to use for standardization.

    Return
    ------
    HubbleFit object
    """
    return HubbleFit(data, corr=corr)



def onflight_model(corr):
    """ builds on the flight a Object corresponding to the correction you want.
    
    This returns this object
    Returns
    -------
    Child of ModelStandardization (with set  STANDARDIZATION)
    """
    class ModelStandardization_mystand( ModelStandardization ):
        STANDARDIZATION = corr

    return ModelStandardization_mystand()

###############################
#                             #
#   Main CLASSES              #
#                             #
###############################
class HubbleFit( BaseFitter, DataSourceHandler ):
    """ """
    PROPERTIES         = ["model",]
    SIDE_PROPERTIES    = ["pec_velocity"]
    DERIVED_PROPERTIES = ["sndata"]


    # =============== #
    #  Main Methods   #
    # =============== #
    def __init__(self,data, corr, empty=False,
                 add_zerror=True, add_lenserr=True):
        """  low-level class to enable to fit a bimodal model on data
        given a probability of each point to belong to a group or the other.

        Parameters
        ----------
        data: [array]
            This dictionary should have this format:
            {name: {k: value1, k.err: err1,
                    k2: value2, k2.err, err2,
                    k3: value3,
                    cov_kk2: cov_between_k_and_k2,
                    cov_k2k: cov_between_k_and_k2,
                    etc.
                   }
            empty entries will be assumed to be 0

            Requested parameters:
            - mag
            - zcmb

            Any list of k, k1,k2 could be used for standardization except
            the requested parameters (mag, zcmb)

        corr: [list of string] -optional-
            List and k-values (see data) that you want to use for standardization.
            
        add_lenserr,add_zerror: [bool] -optional-
            Include the peciluar dispersion/redshift error (add_zerror)
            and lensing error (add_zerror) on the covariance matrix.
            NB: lensing error is set to 0.055*z (Conley et al. 2011, Betoule et al. 2014)
            
        Return
        -------
        Defines the object
        """
        self.__build__()
        if empty:
            return
        
        self.set_data(data)
        # -- for the fit
        # use_minuit has a setter
        self.set_model(make_new_class(corr))
        self.build_sndata(add_zerror=add_zerror, add_lenserr=add_lenserr)

    
    # ---------- #
    #  BUILDER   #
    # ---------- #

    
    def build_sndata(self, add_zerror=True, add_lenserr=True):
        """ """
        self.sndata["zcmb"] = self.get("zcmb")
        self.sndata["mag"]  = self.get("mag")
        self.sndata["corrections"] = np.asarray([self.get(param) for param in self.model.STANDARDIZATION])
        covmat_init = np.zeros((self.npoints, self.model.nstandardization_coef, self.model.nstandardization_coef))
        
        # - diag error
        for i,name in enumerate(["mag"]+self.model.STANDARDIZATION):
            covmat_init[:self.npoints,i,i] = self.get("%s.err"%name, default=0) **2
            
        # - off diag error
        for i,name1 in enumerate(["mag"]+self.model.STANDARDIZATION):
            for j,name2 in enumerate(["mag"]+self.model.STANDARDIZATION):
                if j==i:
                    continue
                covmat_init[:self.npoints,i,j] = covmat_init[:self.npoints,i,j] = \
                  self.get("cov_%s%s"%(name1,name2), default=0)
            
        # - set it
        self.set_covariance_matrix(covmat_init, add_zerror=add_zerror, add_lenserr=add_lenserr)

    # -------- #
    #  FITTER  #
    # -------- #
    # This is only there for the intrinsic stuff
    def fit(self, intrinsic=0,
            use_minuit=None, kfold=None, nsamples=1000,**kwargs):
        """ """
        self.model.set_intrinsic(intrinsic)
        return super(HubbleFit,self).fit(**kwargs)
    
        
    # -------- #
    #  GETTER  #
    # -------- #
    def get_modelchi2(self, parameters):
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
        return -2*self.model.get_loglikelihood(self.sndata["zcmb"], self.sndata["mag"],
                               self.sndata["corrections"], self.sndata["covmatrix"],
                               parameters=parameters)
        
    # -------- #
    #  SETTER  #
    # -------- #
    
    #  Covariance Matrix  
    # ------------------- 
    def set_covariance_matrix(self, cov, add_zerror=True, add_lenserr=True):
        """
        add_lenserr: 0.055*z based on Betoule 2014, Conley 2011
        """
        covmat = cov.copy()
        if add_zerror:    
            self.add_to_covmatrix(covmat, self.systerror_redshift_doppler**2)
        if add_zerror:
            self.add_to_covmatrix(covmat, 0.055*self.get("zcmb"))
                
        self.sndata["covmatrix"] = covmat

    def add_to_covmatrix(self, covmat, value):
        """ add a diagonal term to the covariance
        For instance redshift error, intrinsic dispsersion, weak leasing
        """
        if not hasattr(value,"__iter__"):
            for i in range(self.npoints):
                covmat[i][0][0] += value
        else:
            for i in range(self.npoints):
                covmat[i][0][0] += value[i]
                
        return covmat

    
    # Peculiar Velocity
    # ------------------- 
    def set_peculiar_velocity(self, velocity_km_s):
        """ Set the peculiar velocity that should be used
        for the data.
        
        Parameters
        ----------
        velocity_km_s:
            None:  this will use the default PECULIAR_VELOCITY
            float: this will be the same for every objects (could be 0)
            array: this array must have the size of the data. Each point
                   could then have its own peculiar velocity
        Returns
        -------
        Void
        """
        if velocity_km_s is None:
            velocity_km_s = PECULIAR_VELOCITY

        if hasattr(velocity_km_s,"__iter__"):
            if len(velocity_km_s) != self.npoints:
                raise ValueError("velocity_km_s must have the same size of the number of data (%d vs. %d). It could otherwise be a single float"%(len(velocity_km_s),self.npoints))
            else:
                velocity_km_s = np.asarray(velocity_km_s)

        # - Set it: Float or numpy array
        self._side_properties["pec_velocity"] = velocity_km_s


    # --------- #
    #  PLOTTER  #
    # --------- #
    

    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def peculiar_velocity(self):
        """ Peculiar velocity of galaxy added to the
        magnitude errors
        """
        if self._side_properties["pec_velocity"] is None:
            self.set_peculiar_velocity(None)
            
        return self._side_properties["pec_velocity"]

    # - Derived Properties
    @property
    def sndata(self):
        """ This property is a dictionary containing all necessary data """
        if self._derived_properties["sndata"] is None:
            self._derived_properties["sndata"] = {}
            self._derived_properties["sndata"]["zcmb"] = None
            self._derived_properties["sndata"]["mag"] = None
            self._derived_properties["sndata"]["corrections"] = None
            self._derived_properties["sndata"]["covmatrix"] = None
            
        return self._derived_properties["sndata"]
    
    @property
    def systerror_redshift_doppler(self):
        """ systematic magnitude error caused by errors or z and galaxy peculiar motion"""
        dmp = self.get("zcmb.err")**2 + (self.peculiar_velocity/CLIGHT_km_s)**2
        return  5/np.log(10) * np.sqrt(dmp)/self.get("zcmb")
    
# ========================= #
#                           #
#     Hubblizer             #
#                           #
# ========================= #
class ModelStandardization( BaseModel ):
    """ Virtual Class To Handle Any Standardization """
    STANDARDIZATION= []
    # - created on the flight see __new__
    #FREEPARAMETERS_STD = ["a%d"%(i+1) for i in range(len())]
    #FREEPARAMETERS     = ["M0"]+FREEPARAMETERS_STD

    
    PROPERTIES         = ["cosmo","standard_coef"]
    SIDE_PROPERTIES    = ["sigma_int"]
    DERIVED_PROPERTIES = []

    
    # ================ #
    #  Main Method     #
    # ================ #
    def __new__(cls,*arg,**kwarg):
        """ Black Magic to allow Generalization of Models """
        
        cls.FREEPARAMETERS_STD = ["a%d"%(i+1) for i in range(len(cls.STANDARDIZATION))]
        cls.FREEPARAMETERS     = ["M0"]+cls.FREEPARAMETERS_STD
        return super(ModelStandardization,cls).__new__(cls)
        
    # -------------------- #
    #  Modefit Generic     #
    # -------------------- #
    def set_intrinsic(self, intrinsic_disp):
        """ """
        if intrinsic_disp<0:
            raise ValueError("intrinsic_disp have to be positive or null")
        self._side_properties["sigma_int"] = intrinsic_disp
        
    def setup(self, parameters):
        """ fill the standardization_coef property that will be used for the standardization """
        for name,v in zip(self.FREEPARAMETERS, parameters):
            self.standardization_coef[name] = v

    def get_model(self, z, corrections):
        """ Get the magnitude that should be compared to the observed one:
        example:
           m_obs = 5*log10( D_L ) - alpha*(stretch-1) - beta*colour + scriptm
           (beta will hence be negative)
           return m_obs
         """
        # -- correction alpha*stretch + beta*color
        if corrections is None:
            mcorr = 0
        else:
            mcorr = np.sum([ self.standardization_coef[alpha]*coef
                            for alpha,coef in zip(self.FREEPARAMETERS_STD,corrections)],
                            axis=0)
            
        # - model
        return self.cosmo.distmod(z).value + self.standardization_coef["M0"] + mcorr

        
    def get_loglikelihood(self, z, mag, corrections, covmatrix, parameters=None):
        """ The loglikelihood (-0.5*chi2) of the data given the model

        for N data with M correction (i.e. if x_1 and c standardization, M=2 )
        Parameters
        ----------
        z, mag: [N-array,N-array]
            redshift and observed magnitude of the supernovae

        correction: [NxM array]
            correction parameter for each SNe

        covmatrix: [NxM+1xM+1 matrix]
            +1 because of M0
            The full covariance matrix between the standardization parameters and M0

        // Change the model

        parameters: [array] -optional-
            Change the current model with this parameter setup
        
        Returns
        -------
        float (-0.5*chi2)
        """
        if parameters is not None:
            self.setup(parameters)

        res = mag - self.get_model(z, corrections)
        return -0.5 * np.sum(res**2 / self.get_variance(covmatrix) )

    def lnprior(self,parameter):
        """ so far a flat prior """
        for name_param,p in zip(self.FREEPARAMETERS, parameter):
            if "sigma" in name_param and p<0:
                return -np.inf
        return 0
    
    # -------------------- #
    #  Model Special       #
    # -------------------- #
    def get_variance(self, covmatrix):
        """ This return the variance associated to the given covariance matrix
        taken into account the current alpha, beta etc parameters. It also includes
        the intrinsic dispersion if "sigma_int" indeed is a one of the free parameter
        Returns
        -------
        array (variances)
        """
        #return 1.
        a_ = np.matrix(np.concatenate([[1.0],[self.standardization_coef[k]
                                      for k in self.FREEPARAMETERS_STD]]))
        
        return np.array([np.dot(a_, np.dot(c, a_.T)).sum() for c in covmatrix]) +\
          self.intrinsic_dispersion**2 


    def set_cosmo(self, cosmo):
        """ """
        if astropy.cosmology.core.Cosmology not in cosmo.__class__.__mro__:
            raise TypeError("Only Astropy Cosmology object supported")
        
        self._properties["cosmo"] = cosmo
        
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def intrinsic_dispersion(self):
        if self._side_properties["sigma_int"] is None:
            self.set_intrinsic(0)
        return self._side_properties["sigma_int"]
    
    @property
    def standardization_coef(self):
        """ """
        if self._properties["standard_coef"] is None:
            self._properties["standard_coef"] = {}
        return self._properties["standard_coef"]

    @property
    def nstandardization_coef(self):
        """ Number of standardization parameter +1 (the magnitude of the SN)"""
        return len(self.STANDARDIZATION) + 1
        
    @property
    def cosmo(self):
        """ """
        if self._properties["cosmo"] is None:
            warnings.warn("Using Default Cosmology")
            self.set_cosmo(DEFAULT_COSMO)
            
        return self._properties["cosmo"]
        
