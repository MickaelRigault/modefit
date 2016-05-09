#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" This low level module manages the scipy / minuit variation """

import numpy as np
import warnings

try:
    from iminuit import Minuit
    _HASMINUIT = True
except ImportError:
    warnings.warn("iminuit not accessible. You won't be able to use minuit functions", ImportError)
    __HASMINUIT = False

    
# ========================================== #
#                                            #
# = Use the Scipy-Minuit Tricks            = #
#                                            #
# ========================================== #

class VirtualFitter( object ):
    """
    = Virtual function the real Fitter technic shall inherit = 
    """
    # ========================= #
    # = Fit                   = #  
    # ========================= #
    def fit(self,use_minuit=None,
            **kwargs):
        """
        fit the data following the model.

        Parameters:
        -----------
        
        use_minuit: [bool/None]    If None, this will use the object's current
                                   *use_minuit* value.
                                   If bool, this will set the technique used to
                                   fit the *model* to the *data* and will thus
                                   overwrite the existing *self.use_minuit* value.
                                   Minuit is the iminuit library. Scipy.minimize is the
                                   alterntive technique.
                                   The internal functions are made such that none of
                                   the other function the user have access to depend
                                   on the choice of fitter technique. For instance the
                                   fixing value concept (see set_guesses) remains with
                                   scipy.
                                   
        **kwargs                   goes to self.set_guesses.
                                   (Define the guess, fixed and boundaries values)

        Returns:
        --------
        Void, create output model values.
                    
        """
        self.setup_guesses(**kwargs)
        
        if use_minuit is not None:
            self.use_minuit = use_minuit

        if self.use_minuit:
            self._fit_minuit_()
        else:
            self._fit_scipy_()

        self._fit_readout_()
        
    def _fit_readout_(self):
        """ Gather the output in the readout, you could improve that in you child class"""
        
        self.fitvalues = {}
        for i,name in enumerate(self.model.freeparameters):
            self.fitvalues[name] = self._fitparams[i]
            self.fitvalues[name+".err"] = np.sqrt(self.covMatrix[i,i])
            
        # -- Additional data -- #
        self.fitvalues["chi2"]    = self.get_fval()

        
    # --------------------- #
    # -- Ouput Returns   -- #
    # --------------------- #
    def get_fitvalues(self):
        """
        """
        if not self.fitperformed:
            raise AttributeError("You should fit first !")
        return self.fitvalues.copy()

    def get_fval(self):
        """
        """
        if not self.fitperformed:
            raise AttributeError("You should fit first !")
        
        if self.use_minuit:
            return self.minuit.fval
        return self.scipy_output['fun']

    def converged_on_boundaries(self):
        """Check if any parameter have converged on any boundary of the model"""
        
        if not self.fitperformed:
            raise AttributeError("You should fit first !")

        for name in self.model.freeparameters:
            lowerbound,higherbound = eval("self.model.%s_boundaries"%name)
            if lowerbound is not None and \
              "%.4e"%self.fitvalues[name] == "%.4e"%lowerbound :
                return True
            if higherbound is not None and \
              "%.4e"%self.fitvalues[name] == "%.4e"%higherbound :
                return True
        return False

    def is_fit_good(self):
        return self.fitOk and not self.converged_on_boundaries()
        
        
    # ========================= #
    # = Main Methods          = #
    # ========================= #
    def setup_guesses(self,**kwargs):
        """
        = Defines the values' guesses, boundaries and fixes that will be
          passed to the given model.
          For each variable `v` of the model (see model.freeparameters)
          the following entries will be defined: v_guess, v_boundaries, v_fixed.
          An internal dictionnary _guesses will be created as well as three
          arrays: self.paramguess, self.parambounds,self.paramfixed.
          Those will then be used by the fitter.
        =

        **kwargs                    fill the v_guess, v_boundaries and, v_fixed
                                    for as many `v` as you like.
                                    `v` must be a model's freeParameter.
                                    All the non-given values will be fill either
                                    by pre-existing values in the model or with
                                    0 for _guess, False for _fixed, and
                                    [None,None] for _boundaries

        = RETURNS =
        Void, defines the paramguess, parambounds and paramfixed
        """
        def _test_it_(k,info):
            param = k.split(info)[0]
            if param not in self.model.freeparameters:
                raise ValueError("Unknown parameter %s"%param)

            
        # -- In there, all the model already has
        if "_guesses" not in dir(self):
            self._guesses = self.model.get_given_parameterInfo()
        
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
            self._guesses[k] = v

        # -- Finally if no values have been set, let's do it
        for name in self.model.freeparameters:
            if name+"_guess" not in self._guesses:
                self._guesses[name+"_guess"] = 0
            if name+"_fixed" not in self._guesses:
                self._guesses[name+"_fixed"] = False
            if name+"_boundaries" not in self._guesses:
                self._guesses[name+"_boundaries"] = [None,None]
            # -- and give it to the model
            for v in ["_guess","_fixed","_boundaries"]:
                self.model.__dict__[name+v] = self._guesses[name+v]
                
        # -- Create the paramguess, paramfixed and parambounds
        self._load_guesses_and_associate_()
        
    def _load_guesses_and_associate_(self):
        """
        """
        # -- For the user to follow
        self.paramguess  = [self._guesses["%s_guess"%name]
                        for name in self.model.freeparameters]
        self.paramfixed  = [self._guesses["%s_fixed"%name]
                        for name in self.model.freeparameters]
        self.parambounds = [self._guesses["%s_boundaries"%name]
                        for name in self.model.freeparameters]


    # ==================== #
    # = Bayes & MCMC     = #
    # ==================== #
    # to be finished
    def run_mcmc(self,nrun=2000, walkers_per_dof=3):
        """
        """
        try:
            import emcee
        except ImportError:
            raise ImportError("Install emcee first => sudo pip install emcee")
        
        # -- set up the mcmc
        self.mcmc["ndim"], self.mcmc["nwalkers"] = \
          self.model.nparam, self.model.nparam*walkers_per_dof
        self.mcmc["nrun"] = nrun
        
        # -- init the walkers
        fitted = np.asarray([self.fitvalues[name] for name in self.model.freeparameters])
        err    = np.asarray([self.fitvalues[name+".err"] for name in self.model.freeparameters])
        self.mcmc["pos_init"] = self._fitparams
        self.mcmc["pos"] = [self._fitparams + np.random.randn(self.mcmc["ndim"])*err for i in range(self.mcmc["nwalkers"])]
        # -- run the mcmc        
        self.mcmc["sampler"] = emcee.EnsembleSampler(self.mcmc["nwalkers"], self.mcmc["ndim"], self.model.lnprob)
        _ = self.mcmc["sampler"].run_mcmc(self.mcmc["pos"], self.mcmc["nrun"])

    def set_mcmc_burnin(self, burnin):
        """ Set the burnin value above which the walkers are consistants.
        This is required to access the `samples`
        """
        if burnin<0 or burnin>self.mcmc["nrun"]:
            raise ValueError("the mcmc burnin must be greater than 0 and lower than the amount of run.")
        
        self.mcmc["burnin"] = burnin
        
    def show_mcmc_corner(self, savefile=None, show=True,
                         truths=None,**kwargs):
        """
        **kwargs goes to corner.corner
        """
        try:
            import corner
        except ImportError:
            raise ImportError("install corner to be able to do this plot => sudo pip install corner.")
        from astrobject.utils.mpladdon import figout
        
        fig = corner.corner(self.mcmc_samples, labels=self.model.freeparameters, 
                        truths=self.mcmc["pos_init"] if truths is None else truths,
                        show_titles=True,label_kwargs={"fontsize":"xx-large"})

        fig.figout(savefile=savefile, show=show)
        
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
        # -- This show the 
        import matplotlib.pyplot as mpl
        from astrobject.utils.mpladdon import figout
        if not self.has_mcmc_ran():
            raise AttributeError("you must run mcmc first")
        
        fig = mpl.figure(figsize=[7,3*self.mcmc["ndim"]])
        # -- inputs
        if cline is None:
            cline = mpl.cm.Blues(0.4,0.8)
        if cwalker is None:
            cwalker = mpl.cm.binary(0.7,0.2)
        
        # -- ploting            
        for i, name, fitted in zip(range(self.mcmc["ndim"]), self.model.freeparameters, self.mcmc["pos_init"] if truths is None else truths):
            ax = fig.add_subplot(self.mcmc["ndim"],1,i+1, ylabel=name)
            _ = ax.plot(np.arange(self.mcmc["nrun"]), self.mcmc["sampler"].chain.T[i],
                        color=cwalker,**kwargs)
            
            ax.axhline(fitted, color=cline, lw=2)

        fig.figout(savefile=savefile, show=show)    
    

    # ==================== #
    # = Minuit           = #
    # ==================== #
    def _fit_minuit_(self,verbose=True):
        """
        """
        self._setup_minuit_()
        print "STARTS MINUIT FIT"
        self._migrad_output_ = self.minuit.migrad()
        
        if self._migrad_output_[0]["is_valid"] is False:
            print "** WARNING ** migrad is not valid"
            self.fitOk = False
        elif verbose:
            self.fitOk = True
            
        self._fitparams = np.asarray([self.minuit.values[k]
                              for k in self.model.freeparameters])
        if self._migrad_output_[0]["is_valid"]:
            self.covMatrix    = self.model.read_hess(np.asarray(self.minuit.matrix()))
        else:
            fakeMatrix = np.zeros((len(self._fitparams),len(self._fitparams)))
            for i,k in enumerate(self.model.freeparameters):
                fakeMatrix[i,i] = self.minuit.errors[k]**2
            print "*WARNING* Inaccurate covariance Matrix. Only trace defined"
            self.covMatrix    = self.model.read_hess(fakeMatrix)

        
    def _setup_minuit_(self):
        """
        """
        if "_guesses" not in dir(self):
            self.setup_guesses()
        
        # == Minuit Keys == #
        minuit_kwargs = {}
        for param in self.model.freeparameters:
            minuit_kwargs[param]           = self._guesses["%s_guess"%param]
            minuit_kwargs["limit_"+param]  = self._guesses["%s_boundaries"%param]
            minuit_kwargs["fix_"+param]    = self._guesses["%s_fixed"%param]
            
        self.minuit = Minuit(self.model._minuit_chi2_,
                             print_level=1,errordef=1,
                             **minuit_kwargs)

    # ====================== #
    # = Scipy              = #
    # ====================== #
    def _fit_scipy_(self):
        """
        """
        from scipy.optimize import minimize
        self._setup_scipy_()
        self.scipy_output = minimize(self.model._scipy_chi2_,
                                    self._paramguess_scipy,bounds=self._parambounds_scipy)
        if "hess_inv" in self.scipy_output:
            self.covMatrix     = self.model.read_hess(self.scipy_output['hess_inv'])
        else:
            fakeHess = np.zeros((len(self.scipy_output["x"]),len(self.scipy_output["x"])))
            for i in range(len(fakeHess)):
                fakeHess[i,i]=self.scipy_output["jac"][i]
            self.covMatrix     = self.model.read_hess(fakeHess)
                
        self._fitparams = self.model._read_scipy_parameter_(self.scipy_output["x"])
        self.fitOk         = True
        
    def _setup_scipy_(self):
        """
        = This function manage the fixed values and how
          the parametrisation is made
        =
        """
        self._paramguess_scipy, self._parambounds_scipy = \
          self.model._parameter2scipyparameter_(self.paramguess,self.parambounds)
    
    
    # ====================== #
    # = Properties         = #
    # ====================== #
    @property
    def fitperformed(self):
        return "fitvalues" in dir(self)

    @property
    def mcmc(self):
        """ dictionary containing the mcmc parameters """
        if "_mcmc" not in dir(self):
            self._mcmc = {}
        return self._mcmc

    @property
    def mcmc_samples(self):
        """ the flatten samplers after burned in removal, see set_mcmc_samples """
        if not self.has_mcmc_ran():
            raise AttributeError("run mcmc first.")
        if "burnin" not in self.mcmc.keys():
            raise AttributeError("You did not specified the burnin value. see 'set_mcmc_burnin")
        return self.mcmc["sampler"].chain[:, self.mcmc["burnin"]:, :].reshape((-1, self.mcmc["ndim"]))
        
    def _set_mcmc_(self,mcmcdict):
        """ Advanced methods to avoid rerunning an existing mcmc """
        self._mcmc = mcmcdict
        
    def has_mcmc_ran(self):
        """ return True if you ran 'run_mcmc' """
        return "sampler" in self.mcmc.keys()
    
# ========================================== #
#                                            #
# = Play with Scipy and Minuit Similarly   = #
#                                            #
# ========================================== #

class ScipyMinuitModel ( object ):
    """
    """
    def __init__(self):
        """
        """
        self._checkup_()
        
    # ------------------ #
    # -- Check Class   - #
    # ------------------ #
    def _checkup_(self):
        """ Check if your class has the good parameters """
        
        if "freeparameters" not in dir(self):
            raise AttributeError("The object must have `freeparameters` defined.")
        
    # =================== #
    # = Fitter Methods  = #
    # =================== #
    def setup(self,parameter):
        """ parses the parameter to feed the model class """
        for name,p in zip(self.freeparameters,parameter):
            self.__dict__[name] = p 

    def get_given_parameterInfo(self):
        """
        """
        infodico = {}
        for name in self.freeparameters:
            for info in ["_guess","_fixed","_boundaries"]:
                if name+info in dir(self):
                    infodico[name+info] = eval("self.%s"%(name+info))
        return infodico

    def read_hess(self,hess):
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

    # ==================== #
    # = Bayesian         = #
    # ==================== #
    # assumes you have a `get_chi2(parameters)` methods
    # and chi2 = -2 * logLikelihood
    def lnprior(self,parameters, verbose=True):
        """ perfectely flat prior, should be change by inheriting classed"""
        if verbose: print "Perfectly flat prior used. Always 0 (set verbose=False to avoid this message)"
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
    # = Minuit           = #
    # ==================== #
    # -- useful for the fitter
    def _minuit_chi2_(self,*args,**kwargs):
        """
        """
        raise NotImplementedError(" _minuit_fit_ must be defined in the child function")
    
    
    # ==================== #
    # = Scipy            = #
    # ==================== #
    # -- Scipy shaped like a minuit 
    def _scipy_chi2_(self,parameter):
        """
        """
        parameter = self._read_scipy_parameter_(parameter)
        return self.get_chi2(parameter)

    def _read_scipy_parameter_(self,parameter):
        """
        = This works opposingly to _parameter2scipyparameter_
          it fills the missing values (the fixed one) with the
          value of the model
        =

        parameter: [array]          the scipy-shade parameter
        
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


    # ==================== #
    # = Properties       = #
    # ==================== #
    @property
    def nparam(self):
        return len(self.freeparameters)
    
