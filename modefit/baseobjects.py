#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Low level module to manage the scipy/minuit code variations """

import warnings
import numpy as np

from scipy      import stats

# - astrobject dependencies
try:
    from propobject import BaseObject
except ImportError:
    raise ImportError("You need to install propobject: pip install propobject")
try:
    from iminuit import Minuit    
except ImportError:
    raise ImportError("You need to install iminuit: pip install iminuit")


from .utils     import make_method


###################################
#                                 #
#   Markov Chain Monte Carlo      #
#                                 #
###################################

class MCMC( BaseObject ):
    """ Class Gathering the MCMC run output. Based on emcee """
    

    PROPERTIES         = ["lnprob","freeparameters","runprop",
                          "burnin","properties"]
    SIDE_PROPERTIES    = ["boundaries_poswalkers"]
    DERIVED_PROPERTIES = ["sampler","poswalkers", "chain"]
    
    # MCMC global variables
    RUN_PROPERTIES = ["guess","guess_err","nrun","nwalkers"]
    
    # ========================= #
    #   Initialization          #
    # ========================= #
    def __init__(self, lnprob=None, freeparameters=None,
                 guess=None,guess_err=None, empty=False,
                 boundaries_poswalkers=None):
        """ The mcmc object

        Parameters
        ----------
        lnprob: [function]
            function that must returns the log of log_prior + log_likelihood
            it should take parameters as input.

        freeparameters: [string-array]
            name of parameters of the models.
            (same size and sorting as the parameters entering lnprob()

        guess: [float-array] -optional-
            initial position for the mcmc-walkers

        guess_err: [float-array] -optional-
            typical error range for the initial guess. The walkers will be
            initialized around 'guess' +/- 'guess_err'

        Return
        ------
        Loads the instance.
        """
        if empty:
            return
        
        self._properties["lnprob"] = lnprob
        self._properties["freeparameters"] = freeparameters
        self._side_properties["boundaries_poswalkers"] = boundaries_poswalkers
        
    # ========================= #
    #   Main Methods            #
    # ========================= #
    def run(self, verbose=True,**kwargs):
            
        """ run the mcmc. This method could take time
        (running method based on emcee)
        
        **kwargs could be any `properties` entry:
           nrun, nwalkers, guess, guess_err

        The entry will change the way the mcmc will be performed

        Returns
        -------
        Void
        """
        try:
            import emcee
        except ImportError:
            raise ImportError("Install emcee first => sudo pip install emcee")
        
        self.setup(reset=False,**kwargs)
        if not self.is_setup():
            raise AttributeError("At least one of the following proprety" +\
                                 " has not been setup: nrun, nwalkers, guess, guess_err")
        self.reset(reset_property=False)
        
        # -- run the mcmc        
        self._derived_properties["sampler"] = emcee.EnsembleSampler(self.nwalkers, self.nparam, self.lnprob)
        if verbose:
            print("-> emcee EmsembleSampler defined")
            
        _ = self.sampler.run_mcmc(self.poswalkers, self.nrun)
        
        if verbose:
            print("-> MCMC sampler.run_mcmc() done")
            
    # ------------ #
    # - SETTER   - #
    # ------------ #
    def setup(self, reset=True,**kwargs):
        """ setup one or several entries.
        The derived parameters (sampler, samples) will be reset except
        if reset is set to False

        **kwargs could be any of the RUN_PROPERTIES (nrun, nwalkers, guess, guess_err)
        """

        for k,v in kwargs.items():
            if k not in self.properties.keys():
                raise ValueError("unknown property to setup %s"%k)
            if k in ["guess","guess_err"] and len(v) !=self.nparam:
                raise ValueError("'%s' must be a %d-value array (%d given)"%(k,self.nparam, len(v)))
            self.properties[k] = v
            
        # -- you changed the mcmc, the former values are not correct anymore
        if reset:
            self.reset(reset_property=False)
        
    def set_burnin(self, value):
        """ defines the burnin value above which the walkers
        are consistants. This is required to access the `samples`
        """
        
        if value<0 or value>self.nrun:
            raise ValueError("the mcmc burnin must be greater than 0 and lower"+\
                             " than the amount of run.")
        self._properties["burnin"] = value

    def reset(self, reset_property=True):
        """ clean the derived values """
        if reset_property:
            self._properties["properties"]     = None
            
        self._properties["burnin"]             = None
        self._derived_properties["poswalkers"] = None
        self._derived_properties["sampler"]    = None

    # ------------ #
    # - I/O      - #
    # ------------ #
    def load_data(self, mcmcdata):
        """ load the instance based on the mcmc.data dictionary

        Parameters
        ----------
        mcmcdata: [dict]
            Dictionary as create by the MCMC class' method 'data', containing the keys:
            "chain", "freeparameters", "burnin", "poswalkers"
        """
        # --------------
        # - Input Test
        if type(mcmcdata) is not dict:
            raise TypeError("mcmcdata must be a dict as created by the data method")
        for k in ["chain", "freeparameters", "burnin", "guess"]:
            if k not in mcmcdata.keys():
                raise TypeError("mcmcdata dict must contain the %s key"%k)
        # --------------
        # - Setting
        self._properties["freeparameters"] = mcmcdata["freeparameters"]
        self.set_chain(mcmcdata["chain"])
        self.set_burnin(mcmcdata["burnin"])
        self.properties["guess"] = mcmcdata["guess"]
        
    def set_chain(self, chain):
        """ set the mcmc chain, so you do not need to run_mcmc

        Parameters
        ----------
        chain: [3D N-length array]
            chain containing the walkers information. It's shape is the following:
            [nwalkers, nrun, n-freeparameters], where
               - nwalkers is the number of walkers
               - nrun is the number of mcmc run
               - n-freeparameters is the number of free parameters of the model
               
        Return
        ------
        Void
        """
        # --------------
        # - Input
        try:
            nwalkers, nruns, nfreeparameters = np.shape(chain)
        except:
            raise TypeError("The chain must have the following shape:"+\
                            " (nwalkers, nruns, nfreeparameters)")
        if nfreeparameters != self.nparam:
            raise TypeError("The chain do not have the good number of freeparameter")
        
        if nwalkers<nfreeparameters:
            raise TypeError("The chain must have the following shape:"+\
                            " (nwalkers, nruns, nfreeparameters)."+\
                            " nwalkers must be greater than nfreeparameters")
        # --------------
        # - Input
        self.nwalkers = nwalkers
        self.properties["nrun"] = nruns
        self._derived_properties['chain'] = chain
        
    # ========================= #
    #   Plot Methods            #
    # ========================= #
    # ---------- #
    #  Walkers   #
    # ---------- #
    def show_walkers(self,savefile=None, show=True,
                        cwalker=None, cline=None, truths=None, **kwargs):
        """ Show the walker values for the mcmc run.

        Parameters
        ----------

        savefile: [string]
            where to save the figure. if None, the figure won't be saved

        show: [bool]
            If no figure saved, the function will show it except if this is set
            to False

        cwalker, cline: [matplotlib color]
            Colors or the walkers and input values.
        """
        # -- This show the 
        import matplotlib.pyplot as mpl
        from .utils import figout
        if not self.has_chain():
            raise AttributeError("you must run mcmc first")
        
        fig = mpl.figure(figsize=[7,3*self.nparam])
        # -- inputs
        if cline is None:
            cline = mpl.cm.Blues(0.4,0.8)
        if cwalker is None:
            cwalker = mpl.cm.binary(0.7,0.2)
        
        # -- ploting
        axes = []
        for i, name, fitted in zip(range(self.nparam), self.freeparameters,
                                   self.guess if truths is None else truths):
            ax = fig.add_subplot(self.nparam,1,i+1, ylabel=name)
            _ = ax.plot(np.arange(self.nrun), self.chain.T[i],
                        color=cwalker,**kwargs)
            
            ax.axhline(fitted, color=cline, lw=2)
            axes.append(ax)
        if self.burnin is not None:
            [ax.axvspan(0,self.burnin,
                        fc=mpl.cm.Reds(0.4,0.1),ec=mpl.cm.Reds(0.8,0.4))
            for ax in axes]
            
        fig.figout(savefile=savefile, show=show)

    # ---------- #
    #  Walkers   #
    # ---------- #
    def show_corner(self, savefile=None, show=True,
                         truths=None,**kwargs):
        """ this matrix-corner plot showing the correlation between the
        parameters.

        Parameters
        ----------
        savefile: [string/None]
            where to save the figure. If None, the plot won't be saved

        show: [bool]
            If the plot is not saved. It is shown except if this is False

        truths: [array]
            Show values as lines through the matrix-axes.
            
        **kwargs goes to corner.corner

        Return
        ------
        Void
        """
        try:
            import corner
        except ImportError:
            raise ImportError("install corner to be able to do this plot => sudo pip install corner.")
        
        from .utils import figout
        
        fig = corner.corner(self.samples, labels=self.freeparameters, 
                        truths=self.guess if truths is None else truths,
                        show_titles=True,label_kwargs={"fontsize":"xx-large"},**kwargs)

        fig.figout(savefile=savefile, show=show)
        
    # ========================= #
    #   Properties              #
    # ========================= #
    
    @property
    def data(self):
        """ dictionary containing the basic mcmc information """
        return {"chain":         self.chain,
                "freeparameters":self.freeparameters,
                "burnin":        self.burnin,
                "guess":         self.guess}
    
    #  MCMC base
    # --------------------
    @property
    def freeparameters(self):
        """ names of the parameters of the model (see lnprob) """
        return self._properties["freeparameters"]
    
    @property
    def nparam(self):
        """ number of parameters """
        return len(self.freeparameters)

    @property
    def properties(self):
        """ dictionary containing the parameters needed to setup the mcmc run"""
        if self._properties["properties"] is None:
            self._properties["properties"] = {}
            for k in self.RUN_PROPERTIES :
                self._properties["properties"][k] = None
                
        return self._properties["properties"]

    def is_setup(self):
        """ Check if properties needed to run the mcmc has been defined """
        for k in self.RUN_PROPERTIES:
            if self.properties[k] is None:
                return False
        return True
    
    @property
    def lnprob(self):
        """ functon returning the log_prior + log_likelihood in the
        Bayesian framework."""
        return self._properties["lnprob"]

    #   MCMC properties
    # --------------------
    @property
    def nrun(self):
        """ number of run for the mcmc"""
        return self.properties["nrun"]
    
    @property
    def nwalkers(self):
        """ number of walkers used to scan the parameter space
        This must be at least twice the number of parameters."""
        return self.properties["nwalkers"]

    @nwalkers.setter
    def nwalkers(self,value):
        """ define the number of walker """
        if int(value)<2*self.nparam:
            raise ValueError("emcee request to have at least twice more walkers than parameters.")
        
        self.properties["nwalkers"] = int(value)

    @property
    def poswalkers(self):
        """ Initial position given to the walkers """
        if self._derived_properties["poswalkers"] is None:
            self._derived_properties["poswalkers"] = self.draw_poswalkers()
            
        return self._derived_properties["poswalkers"]

    def draw_poswalkers(self):
        """ return a generation of initial position for the walker.
        If an initial guess hit the boundaries, it will be redrawn """
        if self._boundaries_poswalkers is None:
            return [np.random.uniform(self.guess-self.guess_err,
                                    self.guess+self.guess_err,size=self.nparam)
                    for i in range(self.nwalkers)]

        bounds = np.asarray(self._boundaries_poswalkers, dtype="float")
        return [np.random.uniform(np.nanmax([self.guess-self.guess_err,bounds.T[0]],axis=0),
                                  np.nanmin([self.guess+self.guess_err,bounds.T[1]],axis=0),
                                  size=self.nparam)
                for i in range(self.nwalkers)]
    
    @property
    def _boundaries_poswalkers(self):
        """ boundaries for the initial guess of the walkers"""
        return self._side_properties["boundaries_poswalkers"]
        
    @property
    def guess(self):
        """ Initial central values for to set the walkers """
        return self.properties["guess"]
        
    @property
    def guess_err(self):
        """ Initial errors around the central values
        for to set the walkers"""
        return self.properties["guess_err"]

    @property
    def burnin(self):
        """ Number of walk below which the walker did not converged. """
        return self._properties["burnin"]
    
    #  Derived Properties
    # --------------------
    @property
    def sampler(self):
        """ the emcee mcmc sampler """
        return self._derived_properties["sampler"]

    @property
    def chain(self):
        """ the sampler chain of the walkers """
        if self._derived_properties['chain'] is None:
            return self.sampler.chain if self.sampler is not None else None
        
        return self._derived_properties['chain']
        
    def has_chain(self):
        """ return True if you ran 'run_mcmc' """
        return self.chain is not None

    @property
    def samples(self):
        """ the flatten samplers after burned in removal, see set_mcmc_samples """
        if not self.has_chain():
            raise AttributeError("run mcmc first.")
        if self.burnin is None:
            raise AttributeError("You did not specified the burnin value. see 'set_burnin")
        
        return self.chain[:, self.burnin:, :].reshape((-1, self.nparam))

    @property
    def nsamples(self):
        """ number of samples avialable (burnin removed) """
        return len(self.samples)


    @property
    def derived_values(self):
        """ 3 times N array of the derived parameters
            [50%, +1sigma (to 84%), -1sigma (to 16%)]
        """
        return map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(self.samples, [16, 50, 84],axis=0)))
    
    @property
    def derived_parameters(self):
        """ dictionary of the mcmc derived values with the structure:
           NAME_OF_THE_PARAMETER = 50% pdf
           NAME_OF_THE_PARAMETER.err = [+1sigma, -1sigma]
        """
        values = self.derived_values
        fitout = {}
        for v,name in zip(values, self.freeparameters):
            fitout[name] = v[0]
            fitout[name+".err"] = [v[1],v[2]]
            
        return fitout

###################################
#                                 #
#   Data Management Classes       #
#                                 #
###################################
# ================ #
#                  #
#  K-Folding       #
#                  #
# ================ #
class _KFolder_( BaseObject ):
    """ Virtual class to be packed inherited by DataHandlers-like ones.
    
    This Virtual class enables to deal with K-folding tools
    """
    PROPERTIES         = []
    SIDE_PROPERTIES    = ["used_indexes"]
    DERIVED_PROPERTIES = ["fold_indexes","kfold"]

    # ==================== #
    #    Main Method       #
    # ==================== #

    # --------------- #
    #  folding      - #
    # --------------- #
    def fold_data(self, kfold, nsamples = 1000):
        """ """
        # ---------- #
        # - inputs - #
        kfold = int(kfold)
        if int(kfold) <2:
            raise ValueError("kfold must be greater than 2 (int), %d given"%(kfold))
        nsamples = int(nsamples)
        if int(nsamples) <1:
            raise ValueError("nsamples must be at least 1 (int), %d given"%(nsamples))

        # ---------- #
        # - inputs - #
        indexes = np.arange(self.npoints)
        noutfold  = int( self.npoints/float(kfold) ) # fold removed
        
        self._derived_properties["fold_indexes"] = []
        for i in range(nsamples):
            np.random.shuffle(indexes)
            self._derived_properties["fold_indexes"].append(indexes[noutfold:].copy())
            

    def run_kfolding(self, kfold, nsamples=1000, **kwargs):
        """ Set the kfold property that contains a copy of the current instance
        with a kfolded fit() applied.
        
        **kwargs goes to fit()
        """
        folded = self.copy()
        folded.fit(kfold=kfold, nsamples=nsamples, **kwargs)
        self._derived_properties["kfold"] = folded

    # ==================== #
    #    Properties        #
    # ==================== #
    @property
    def used_indexes(self):
        """ Indexes of the data used for the fitting """
        if self._side_properties["used_indexes"] is None:
            self._side_properties["used_indexes"] = np.arange(self.npoints)
        return self._side_properties["used_indexes"]
    
    def set_used_indexes(self, indexes):
        """ Indexes of the data used for the fitting """
        self._side_properties["used_indexes"] = indexes

    @property
    def kfold(self):
        """ """
        return self._derived_properties["kfold"]

    def has_kfold(self):
        """ Test if the current instance has a kfold set. True means yes"""
        return self.kfold is not None
    
    @property
    def _foldindexes(self):
        """ list of indexes to use for the folding """
        if self._derived_properties["fold_indexes"] is None:
            warnings.warn("No foilding index defined. returns the list of all indexes")
            self._derived_properties["fold_indexes"] = [np.arange(self.npoints)]
            
        return self._derived_properties["fold_indexes"]

# ================ #
#                  #
#  Data Handlers   #
#                  #
# ================ #

class DataHandler( _KFolder_):
    """ """
    PROPERTIES         = ["data","error"]
    SIDE_PROPERTIES    = ["names"]
    DERIVED_PROPERTIES = []

    # ============== #
    #  Main Methods  #
    # ============== #
    def set_data(self, data, errors=None, names=None):
        """ Basic method to set the data """
        self._properties["data"]   = data
        self._properties["errors"] = errors
        self._properties["names"]  = names
        
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def data(self):
        """ Data used for the fit """
        return self._properties["data"]
    
    @property
    def errors(self):
        """ Errors associated to the data """
        return self._properties["errors"]
    
    @property
    def names(self):
        """ Names associated to the data - if set """
        return self._side_properties["names"]

    @property
    def npoints(self):
        return len(self.data)

    
class DataSourceHandler( _KFolder_ ):
    """ Deal with complex data sources, which are dictionary oriented """
    PROPERTIES         = ["data"]
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []

    # =============== #
    #  Main Methods   #
    # =============== #
    # ---------- #
    #  SETTER    #
    # ---------- #
    def set_data(self, data):
        """ data must be a dictionary with names as entries and values then:
        data = {NAME1:{k1:v11, k2:v21 ....}, NAME2:{k1:v12, k2:v22....}, ...}
        """
        self._properties["data"] = data

    # ---------- #
    #  GETTER    #
    # ---------- #
    def get(self, key, names=None, default=None):
        """ Return the value(s) for the given key
        The value is returned for the list of names. If None, this is will
        be all the known names
        """
        names = self.names if names is None else names
        if hasattr(names,"__iter__"):
            return np.asarray([self.data[name_][key] if key in self.data[name_].keys() else default
                    for name_ in names])
        
        return self.data[names][key] if key in self.data[names].keys() else default
    
    # =============== #
    #  Properties     #
    # =============== #
    @property
    def data(self):
        """ Data used for the fit """
        if self._properties["data"] is None:
            warnings.warn("Empty Data")
            self._properties["data"] = {}
            
        return self._properties["data"]
    
    # Names and Size
    # ---------------
    @property
    def names(self):
        """ """
        return np.sort(self._orig_names)
    
    @property
    def _orig_names(self):
        """ """
        return self.data.keys()

    @property
    def npoints(self):
        """ number of points in data """
        return len(self._orig_names)


    
###################################
#                                 #
#   Basic Fitter Object           #
#                                 #
###################################

class BaseFitter( BaseObject ):
    """ Mother class of the fitters """

    PROPERTIES         = ["param_input","model","use_minuit"]
    SIDE_PROPERTIES    = ["kfold", "nfold"]
    DERIVED_PROPERTIES = ["fitvalues","mcmc"]
    
    # ========================= #
    # = Initialization        = #  
    # ========================= #
    def copy(self, empty=False):
        """ """
        c = super(BaseFitter, self).copy(empty=False)
        c.set_model(self.model.__new__(self.model.__class__))
        return c
    
    # ========================= #
    # = Main Methods          = #  
    # ========================= #
    # ---------------
    # - Da Fit
    def fit(self,use_minuit=None, kfold=None, nsamples=1000,
            **kwargs):
        """
        fit the data following the model.

        Parameters:
        -----------
        use_minuit: [bool/None]
            If None, this will use the object's current *use_minuit* value.
            If bool, this will set the technique used to fit the *model*
            to the *data* and will thus overwrite the existing
            *self.use_minuit* value.
            Minuit is the iminuit library.
            Scipy.minimize is the alterntive technique.
            The internal functions are made such that none of the other function
            the user have access to depend on the choice of fitter technique.
            For instance the fixing value concept (see set_guesses) remains with
            scipy.

        // K Folding

        kfold: [int, None] -optional-
        
        nsamples: [int]

        // Kwargs
        
        **kwargs parameter associated values with the shape:
            'parametername'_guess, 'parametername'_fixed, 'parametername'_boundaries 

        Returns:
        --------
        Void, create output model values.
                    
        """
        if kfold is not None and DataHandler not in self.__class__.__mro__:
            raise ValueError("Only Fitter inherating from DataHandler can use k-folding. Set kfold to None")
        
        self.setup_guesses(**kwargs)
        self._derived_properties["fitvalues"] = None
        
        if use_minuit is not None:
            self.use_minuit = use_minuit

        # --------------
        # - Run the fit
        # --------------
        # => No Folding
        if kfold is None:
            if DataHandler in self.__class__.__mro__:
                self.set_used_indexes(np.arange(self.npoints).copy())
            # Da Fit
            self._fit_(step=kwargs.pop("step",1))
            
        # => Folding
        else:
            self.fold_data(kfold, nsamples=nsamples)
            for i,indexes in enumerate(self._foldindexes):
                self.set_used_indexes(indexes)
                # Da Fit
                self._fit_(step=kwargs.pop("step",1))                
                self.fitvalues.setdefault("id",[]).append(self.used_indexes)

        if DataHandler in self.__class__.__mro__:
            self.set_used_indexes(None)
        self._fit_readout_cleaning_(kfold is not None)

    def _fit_(self, step=1):
        """ """
        if self.use_minuit:
            self._fit_minuit_(step=step)
        else:
            self._fit_scipy_()
        
        self._fit_readout_()

    def get_residuals(self, parameters=None):
        """ the data residuals:
        self.data - self.get_model(parameters) 

        Returns
        -------
        Array
        """
        return self.data - self.get_model(self._fitparams if parameters is None else parameters)
    
        

        
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
        return -2 * self.model.get_loglikelihood(*self._get_model_args_())

    
    # --------------------- #
    # -- Da Model        -- #
    def set_model(self,model,**kwargs):
        """ Load a BaseModel to the fitter.

        Parameters
        -----------
        model: [modefit.BaseModel]
            model use to fit the data. This model must inherit from the mother
            model class BaseModel.

        **kwargs goes to self.set_guesses.
           (Define the guess, fixed and boundaries values)
           
        Returns
        -------
        Void, defines model and associated objects
        """
        self._properties["model"] = model
        try:
            self.model.get_chi2 = self.get_modelchi2
        except:
            raise ValueError(" You must define a 'get_modelchi2()' method in your fitter.")
        self.setup_guesses(**kwargs)

    def get_model(self,parameters):
        """ call the model.get_model() method
        """
        return self.model.get_model(parameters)
    
    # --------------------- #
    # -- Ouput Returns   -- #
    # --------------------- #
    def get_fitvalues(self, mcmc=False, nonsymerrors=False):
        """ get a copy of the fitvalue paramters

        Parameters
        ----------
        mcmc: [bool]
            the parameters estimation made using mcmc.

        nonsymerrors: [bool]
            if the fitted errors are symmetric this converts the '.err'
            entry to an array [err,err] 
        
        Return
        ------
        dictionary (paramname=best_value ; paramname.err = err or [errlow,errup])
        """
        if mcmc and (not self.has_mcmc() or not self.mcmc.has_chain()):
            raise ValueError("No mcmc run yet.")
        
        if not mcmc and not self.has_fit_run():
            raise AttributeError("No fit run yet !")
        # ---------
        # - MCMC
        if mcmc:
            return self.mcmc.derived_parameters.copy()
        # ---------
        # - Fit        
        f = self.fitvalues.copy()
        if nonsymerrors:
            for k,v in f.items():
                if ".err" in k and "__iter__" not in dir(v):
                    f[k] = [v,v]
            
        return f

    def get_fval(self):
        """
        """
        if not self.has_fit_run():
            raise AttributeError("You should fit first !")
        
        return self.minuit.fval if self.use_minuit else \
            self.scipy_output['fun']

    def converged_on_boundaries(self, tested_parameters="all"):
        """Check if any parameter have converged on any boundary of the model"""
        
        if not self.has_fit_run():
            raise AttributeError("You should fit first !")

        for name in self.model.freeparameters:
            if tested_parameters is not "all" and name not in tested_parameters:
                continue
            lowerbound,higherbound = eval("self.model.%s_boundaries"%name)
            if lowerbound is not None and \
              "%.4e"%self.fitvalues[name] == "%.4e"%lowerbound :
                return True
            if higherbound is not None and \
              "%.4e"%self.fitvalues[name] == "%.4e"%higherbound :
                return True
        return False

    def is_fit_good(self):
        """ Test if the fit went well (fitOk good) and if
        no fitted values converged on the boundaries """
        return self.fitOk and not self.converged_on_boundaries()
        
        
    # ========================= #
    # = Main Methods          = #
    # ========================= #
    def setup_guesses(self,**kwargs):
        """ Defines the guesses, boundaries and fixed values
        that will be passed to the given model.

        For each variable `v` of the model (see model.freeparameters)
        the following array will be defined and set to param_input:
           * v_guess,
           * v_boundaries,
           * v_fixed.
        Three arrays (self.paramguess, self.parambounds,self.paramfixed)
        will be accessible that will point to the defined array.

        Parameter
        ---------
        **kwargs the v_guess, v_boundaries and, v_fixed for as many
        `v` (from the model.freeparameter list).
        All the non-given `v` values will be filled either by pre-existing
        values in the model or with: 0 for _guess, False for _fixed, and
        [None,None] for _boundaries

        Return
        ------
        Void, defines param_input (and consquently paramguess, parambounds and paramfixed)
        """
        def _test_it_(k,info):
            param = k.split(info)[0]
            if param not in self.model.freeparameters:
                raise ValueError("Unknown parameter %s"%param)

        # -- In there, all the model already has
        if not self.is_input_set():
            self._properties['param_input'] = self.model.get_param_input()
        
        # -- Then, whatever you gave
        for k,v in kwargs.items():
            if "_guess" in k:
                _test_it_(k,"_guess")
            elif "_fixed" in k:
                _test_it_(k,"_fixed")
            elif "_boundaries" in k:
                _test_it_(k,"_boundaries")
            else:
                raise ValueError("I am not able to parse %s ; not _guess, _fixed nor _boundaries"%k)
            self.param_input[k] = v

        # -- Finally if no values have been set, let's do it
        for name in self.model.freeparameters:
            if name+"_guess" not in self.param_input.keys():
                self.param_input[name+"_guess"] = 0
            if name+"_fixed" not in self.param_input.keys():
                self.param_input[name+"_fixed"] = False
            if name+"_boundaries" not in self.param_input.keys():
                self.param_input[name+"_boundaries"] = [None,None]
            # -- and give it to the model
        self.model.set_param_input(self.param_input)
          
    # ------------------- #
    # - Bayes & MCMC    - #
    # ------------------- #
    def run_mcmc(self,nrun=2000, walkers_per_dof=4,
                 init=None, init_err=None, verbose=True):
        """ run mcmc from the emcee python code. This might take time

        Parameters
        ----------
        nrun: [int]
            number of step run the walkers are going to do.

        walkers_per_dof: [int/float]
            number of walker by degree of freedom (int of nparameter*this used)
            walkers_per_dof should be greater than 2.

        Returns
        -------
        Void (fill the self.mcmc property)
        """
        # -------------
        # - Load MCMC
        self.setup_mcmc(nrun=nrun, walkers_per_dof=walkers_per_dof,
                        init=init, init_err=init_err)
        
        # -------------
        # - And run it
        self.mcmc.run(verbose=verbose)
        
    def setup_mcmc(self, nrun=2000, walkers_per_dof=4,
                 init=None, init_err=None, verbose=True):
        """ Setup the basic property for the mcmc to run, you just need to say self.mcmc.run()
            to run it. See also self.run_mcmc()
        """
        self._derived_properties["mcmc"] = MCMC(self.model.lnprob, self.model.freeparameters,
                                                boundaries_poswalkers=self._mcmc_initbounds)
        
        
        # -------------
        # - Set it up
        guess     = np.asarray([self.fitvalues[name] for name in self.model.freeparameters]) \
                                if init is None else np.asarray(init)
        guess_err = np.asarray([self.fitvalues[name+".err"]
                                for name in self.model.freeparameters]) \
                                if init_err is None else np.asarray(init_err)

        self.mcmc.setup(nrun      = nrun,
                        nwalkers  = walkers_per_dof * self.model.nparam,
                        guess     = guess,
                        guess_err = guess_err)
        
        
    def set_mcmc_burnin(self, burnin):
        """ set the burnin value above which the walkers are consistants.
        This is required to access the `samples`
        """
        self.mcmc.set_burnin(burnin)

    def set_mcmc(self, mcmcdata):
        """ Setup the mcmc with existing data. """
        mcmc = MCMC(empty=True)
        mcmc.load_data(mcmcdata)
        # -- Test
        if mcmc.freeparameters != self.model.freeparameters:
            raise ValueError("the mcmcdata freeparameters do not correspond to that of the model.")
        self._derived_properties["mcmc"] = mcmc
        
    # ==================== #
    #  Ploting Methods     #
    # ==================== #        
    def show_mcmc_corner(self, savefile=None, show=True,
                         truths=None,**kwargs):
        """ this matrix-corner plot showing the correlation between the
        parameters.

        Parameters
        ----------
        savefile: [string/None]
            where to save the figure. If None, the plot won't be saved

        show: [bool]
            If the plot is not saved. It is shown except if this is False

        truths: [array]
            Show values as lines through the matrix-axes.
            
        **kwargs goes to corner.corner

        Return
        ------
        Void
        """
        self.mcmc.show_corner(savefile=savefile, show=show,
                            truths=truths,**kwargs)
        
    def show_mcmcwalkers(self, savefile=None, show=True,
                        cwalker=None, cline=None, truths=None, **kwargs):
        """ Show the walker values for the mcmc run.

        Parameters
        ----------

        savefile: [string]
            where to save the figure. if None, the figure won't be saved

        show: [bool]
            If no figure saved, the function will show it except if this is set
            to False

        cwalker, cline: [matplotlib color]
            Colors or the walkers and input values.
        """
        self.mcmc.show_walkers(savefile=savefile, show=show,
                        cwalker=cwalker, cline=cline, truths=truths, **kwargs)
    
    # ====================== #
    # = Properties         = #
    # ====================== #
    # -------------
    # - Model
    @property
    def model(self):
        """ This model-object used to fit the data """
        return self._properties["model"]
    
    def is_model_set(self):
        """ Test if the model has been set. True means yes """
        return self._properties["model"] is not None
    
    @property
    def dof(self):
        """ return the degree of freedom
        size of the datapoint - number of non-fixed parameters 
        """
        if not hasattr(self, "npoints"):
            raise AttributeError("npoints not define in this model")
        if not self.is_model_set():
            raise AttributeError("No model defined")
        
        return self.npoints - self.model.nparam + np.sum(self.model.paramfixed)
    
    # -----------------------
    # - fit associated values
    @property
    def fitvalues(self):
        """ dictionary containing the best-fitted values """
        if self._derived_properties["fitvalues"] is None:
            self._derived_properties["fitvalues"] = {}
        return self._derived_properties["fitvalues"]

    def has_fit_run(self):
        return len(self.fitvalues.keys())>0

    @property
    def use_minuit(self):
        if self._properties["use_minuit"] is None:
            warnings.warn("No value set for 'use_minuit' => True is set by default")
            self._properties["use_minuit"] = True
        return self._properties["use_minuit"]
    
    @use_minuit.setter
    def use_minuit(self,use_minuit_bool):
        """ set the use_minuit value """
        self._properties["use_minuit"] = bool(use_minuit_bool)
        
    # -----------------------
    # Guess / Boundaries etc
    # -----------------------
    @property
    def param_input(self):
        """ dictionary containing the input parameter values:
           guess / fixed / bounds
        (See also the associated properties)
        """
        if self._properties["param_input"] is None:
            self._properties["param_input"] = {}
        return self._properties["param_input"]
    
    def is_input_set(self):
        """ Test if you set the parameters inputs """
        return len(self.param_input.keys())>0
    
    @property
    def paramguess(self):
        """ guess put for the fit """
        return [self.param_input["%s_guess"%name]
                for name in self.model.freeparameters]
    @property
    def parambounds(self):
        """ guess put for the fit """
        return [self.param_input["%s_boundaries"%name]
                for name in self.model.freeparameters]
    @property
    def _mcmc_initbounds(self):
        """ The mcmc init bounds for a faster mcmc convergence. Defautl parambounds """
        return self.parambounds if "_mcmc_initbounds" not in dir(self.model) else \
          self.model._mcmc_initbounds
        
    @property
    def paramfixed(self):
        """ guess put for the fit """
        return [self.param_input["%s_fixed"%name]
                for name in self.model.freeparameters]

    # ---------------
    # - MCMC Stuffs  
    @property
    def mcmc(self):
        """ MCMC class containing the mcmc parameters """
        return self._derived_properties["mcmc"]

    def has_mcmc(self):
        """ return True if you ran 'run_mcmc' """
        return self.mcmc is not None
    
    @property
    def mcmc_samples(self):
        """ the flatten samplers after burned in removal, see set_mcmc_samples """
        if not self.has_mcmc():
            raise AttributeError("run mcmc first.")
        return self.mcmc.samples

    @property
    def mcmc_fitvalues(self):
        """ dictionary of the mcmc derived values with the structure:
           NAME_OF_THE_PARAMETER = 50% pdf
           NAME_OF_THE_PARAMETER.err = [+1sigma, -1sigma]
        
        """
        if not self.has_mcmc():
            raise AttributeError("run mcmc first.")
        return self.mcmc.derived_values

    @property
    def mcmc_bestparam(self):
        """ best parameter best on the mcmc run. see mcmc_fitvalues for associated errors """
        if not self.has_mcmc():
            raise AttributeError("run mcmc first.")
        return self.mcmc.derived_parameters
    
    # -----------------
    # - derived
    @property
    def covmatrix(self):
        """ coveriance matrix after the fit """
        
        if self.use_minuit:
            if self._migrad_output_[0]["is_valid"]:
                return self.model._read_hess_(np.asarray(self.minuit.matrix()))
            else:
                fakeMatrix = np.zeros((len(self._fitparams),len(self._fitparams)))
                for i,k in enumerate(self.model.freeparameters):
                    fakeMatrix[i,i] = self.minuit.errors[k]**2
                warnings.warn("Inaccurate covariance Matrix. Only trace defined")
                return self.model._read_hess_(fakeMatrix)
        else:
            if "hess_inv" in self.scipy_output:
                return self.model._read_hess_(self.scipy_output['hess_inv'])
            else:
                fakeHess = np.zeros((len(self.scipy_output["x"]),len(self.scipy_output["x"])))
                for i in range(len(fakeHess)):
                    fakeHess[i,i]=self.scipy_output["jac"][i]
                return self.model._read_hess_(fakeHess)

    # ====================== #
    # = Internal Methods   = #
    # ====================== #
    def _fit_readout_(self, kfolding=False):
        """ Gather the output in the readout, you could improve that in you child class"""
        
        for i,name in enumerate(self.model.freeparameters):
            self.fitvalues.setdefault(name,[]).append(self._fitparams[i])
            self.fitvalues.setdefault(name+".err",[]).append(np.sqrt(self.covmatrix[i,i]))
                
        # -- Additional data -- #
        self.fitvalues.setdefault("chi2",[]).append(self.get_fval())

    def _fit_readout_cleaning_(self, folding):
        """ """
        if not folding:
            # no folding so cleaning the fitvalue
            for k,v in self.fitvalues.items():
                self.fitvalues[k] = v[0] if hasattr(v,"__iter__") else v
        else:
            for k,v in self.fitvalues.items():
                self.fitvalues[k] = np.asarray(v)
    # --------------
    #  Minuit
    # --------------    
    def _fit_minuit_(self,verbose=False, step=1):
        """
        """
        self._setup_minuit_(step=step)
        if verbose: print("STARTS MINUIT FIT")
        self._migrad_output_ = self.minuit.migrad()
        
        if self._migrad_output_[0]["is_valid"] is False:
            warnings.warn("migrad is not valid")
            self.fitOk = False
        elif verbose:
            self.fitOk = True
            
        self._fitparams = np.asarray([self.minuit.values[k]
                              for k in self.model.freeparameters])
        
    def _setup_minuit_(self, step=1, print_level=0):
        """
        """
        if "_guesses" not in dir(self):
            self.setup_guesses()
        
        # == Minuit Keys == #
        minuit_kwargs = {}
        for param in self.model.freeparameters:
            minuit_kwargs[param]           = self.param_input["%s_guess"%param]
            minuit_kwargs["limit_"+param]  = self.param_input["%s_boundaries"%param]
            minuit_kwargs["fix_"+param]    = self.param_input["%s_fixed"%param]

        self.minuit = Minuit(self.model._minuit_chi2_,
                             print_level=print_level,errordef=step,
                             **minuit_kwargs)
    # ----------------
    #  Scipy
    # ----------------
    def _fit_scipy_(self):
        """ fit using scipy """
        
        from scipy.optimize import minimize
        self._setup_scipy_()
        self.scipy_output = minimize(self.model._scipy_chi2_,
                                    self._paramguess_scipy,bounds=self._parambounds_scipy)
                
        self._fitparams = self.model._read_scipy_parameter_(self.scipy_output["x"])
        self.fitOk         = True
        
    def _setup_scipy_(self):
        """ manages the fixed values and how the parametrisation is made """
        self._paramguess_scipy, self._parambounds_scipy = \
          self.model._parameter2scipyparameter_(self.paramguess,self.parambounds)


###################################
#                                 #
#   Basic Model Object            #
#                                 #
###################################

class BaseModel( BaseObject ):
    """ Mother class of the Models """

    PROPERTIES         = ["freeparameters"]
    SIDE_PROPERTIES    = ["param_input"]
    DERIVED_PROPERTIES = []
    
    # =================== #
    # = Initialization  = #
    # =================== #
    def __new__(cls,*arg,**kwargs):
        """ Upgrade of the New function to enable the
        the _minuit_ black magic
        """
        obj = super(BaseModel,cls).__new__(cls)
        
        exec("@make_method(BaseModel)\n"+\
             "def _minuit_chi2_(self,%s): \n"%(", ".join(obj.FREEPARAMETERS))+\
             "    parameters = %s \n"%(", ".join(obj.FREEPARAMETERS))+\
             "    return self.get_chi2(parameters)\n")

        exec("@make_method(BaseModel)\n"+\
             "def _minuit_lnprob_(self,%s): \n"%(", ".join(obj.FREEPARAMETERS))+\
             "    parameters = %s \n"%(", ".join(obj.FREEPARAMETERS))+\
             "    return self.lnprob(parameters)\n")

        return obj
    
    # =================== #
    # = Fitter Methods  = #
    # =================== #
    def get_model(self,parameters):
        """ return the model parameter in format of the data
        residual should then by ```data - get_model(parameters)```

        This function is not defined here. do so
        """
        raise NotImplementedError("The Model has no get_model() defined. Do so.")

    def get_param_input(self):
        """ return a pseudo param_input dictionary using the currently
        known parameter information (_guess, _fixed, _boundaries).
        Some might have been set manually when creating the dictionary.
        If so, they will be in the returned dictionary.

        Remark: If you set param_input already, this should be a copy.

        Return
        ------
        dictionary
        """
        infodico = {}
        for name in self.freeparameters:
            for info in ["_guess","_fixed","_boundaries"]:
                if hasattr(self, name+info):
                    infodico[name+info] = eval("self.%s"%(name+info))
        return infodico
        
    # ==================== #
    # = Bayesian         = #
    # ==================== #
    # assumes you have a `get_chi2(parameters)` methods
    # and chi2 = -2 * logLikelihood
    def lnprior(self,parameters, verbose=True):
        """ perfectely flat prior, should be change by inheriting classed"""
        if verbose: print("Perfectly flat prior used. Always 0 (set verbose=False to avoid this message)")
        return 0
        
    def lnprob(self,parameters):
        """ This is the Bayesian posterior function (in log).
        it returns  lnprior - 0.5*Chi2
        (Assuming Chi2 = -2logLikelihood)
        """
        priors = self.lnprior(parameters)
        if not np.isfinite(priors):
            return -np.inf
        # not sure it works with the _minuit_chi2_/_scipy_chi2_  tricks
        return priors - 0.5*self.get_chi2(parameters)

    # ==================== #
    # = Properties       = #
    # ==================== #
    @property
    def freeparameters(self):
        """ freeparameters for the model.
        You could set FREEPARAMETERS in the global argument of a class to uses this.
        """
        if self._properties["freeparameters"] is None:
            from copy import copy
            self._properties["freeparameters"] = copy(self.FREEPARAMETERS)
        return self._properties["freeparameters"]
    
    @property
    def nparam(self):
        return len(self.freeparameters)
    
    #   Parameters Values
    # -----------------------
    @property
    def param_input(self):
        """ dictionary containing the input parameter values:
           guess / fixed / bounds

        See also the respective class properties
        """
        if self._side_properties["param_input"] is None:
            self._side_properties["param_input"] = {}
        return self._side_properties["param_input"]
    
    def set_param_input(self,param_input):
        """ Test if you set the parameters inputs """
        for k in param_input.keys():
            if "_guess" not in k and "_fixed" not in k and "_boundaries" not in k:
                raise ValueError("Parameters input must have information like _guess, _fixed, _boundaries only")
        self._side_properties["param_input"] = param_input
        # -- temporary
        for k,v in self._side_properties["param_input"].items():
            self.__dict__[k] = v
            
    @property
    def paramguess(self):
        """ guess put for the fit """
        return [self.param_input["%s_guess"%name]
                for name in self.freeparameters]
    @property
    def parambounds(self):
        """ guess put for the fit """
        return [self.param_input["%s_boundaries"%name]
                for name in self.freeparameters]
        
    @property
    def paramfixed(self):
        """ guess put for the fit """
        return [self.param_input["%s_fixed"%name]
                for name in self.freeparameters]

    # ==================== #
    # = Internal         = #
    # ==================== #
    def _read_hess_(self,hess):
        """
        """
        if len(hess)==len(self.freeparameters):
            return hess
        
        indexFixed = [i for i,name in enumerate(self.freeparameters)
                      if "%s_fixed"%name in dir(self) and eval("self.%s_fixed"%name)]
        for i in indexFixed:
            newhess = np.insert(hess,i,0,axis=0)
            newhess = np.insert(newhess,i,0,axis=1)
            hess = newhess
            
        return hess

    # -------------------
    #   Scipy
    # -------------------
    # shaped like a minuit 
    def _scipy_chi2_(self,parameter):
        """
        """
        parameter = self._read_scipy_parameter_(parameter)
        return self.get_chi2(parameter)

    def _read_scipy_parameter_(self,parameter):
        """ works opposingly to _parameter2scipyparameter_
        it fills the missing values (the fixed one) with the
        value of the model
        
        parameter: [array]          the scipy-shade parameter

        Return
        ------
        total parameters
        """
        # -- This enable to fasten the code
        if len(parameter) == len(self.freeparameters):
            return parameter
        
        ptotal,ip = [],0
        for name in self.freeparameters:
            if "%s_fixed"%name not in dir(self) or\
               eval("self.%s_fixed"%name) is False:
                ptotal.append(parameter[ip])
                ip +=1
            else:
                ptotal.append(eval("self.%s_guess"%name))
                
        return ptotal
                              
    def _parameter2scipyparameter_(self,guess,bounds,fixed=None):
        """
        = Create what Guess and Bounds scipy must have to account for
          fixed values
        =

        fixed: [array/None]         if you whish to check fixed is correctly
                                    set. Not needed otherwise since this looks
                                    directly to the given `v`_fixed
        """
        scipyGuess,scipyBounds = [],[]
        for name,g,b in zip(self.freeparameters,
                            guess,bounds):
            if "%s_fixed"%name not in dir(self) or\
               eval("self.%s_fixed"%name) is False:
                scipyGuess.append(g)
                scipyBounds.append(b)
                
        return np.asarray(scipyGuess),np.asarray(scipyBounds)
