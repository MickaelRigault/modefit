#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to fit the main optical emission line """
# // version of speclines to test the parameter properties setup as in

import numpy       as np
import warnings

from astropy       import constants
from scipy         import stats

# local dependency
from .baseobjects import BaseModel,BaseFitter

# astrobject dependencies
from astrobject.astrobject.spectroscopy import Spectrum

# - to be removed
from astrobject.utils.decorators import _autogen_docstring_inheritance



__all__ = ["linefit"]



# ============================ #
#                              #
#   GLOBAL INFORMATION         #
#                              #
# ============================ #

CLIGHT=constants.c.to("km/s").value

# - Wavelengths
# http://zuserver2.star.ucl.ac.uk/~msw/lines.html
DICT_EMISSIONLINES = {
    "OII1":3725.5,"OII2":3729,
    "HN":3835.4,"HC":3889.0,
    "HE":3970.,"HD":4101.8,
    "HG":4340.5,"HB":4861.3,
    "OIII_1":4958.9,"OIII_2":5007.,
    "NII_1":6548.04,"HA":6562.8,"NII_2":6583.45,
    "SII1":6716.3,"SII2":6730.7    
    }

# - Sorted Names
LINENAMES = np.asarray(["OII1", "OII2","HN","HC","HE","HD","HG","HB",
                        "OIII_1","OIII_2","HA","NII_1","NII_2",
                        "SII1","SII2"])
LINEWAVES = np.asarray([DICT_EMISSIONLINES[k] for k in
                        LINENAMES])

_NLINES   = len(LINENAMES)
DOUBLETS  = ["NII","OIII"]

# - Index of the Name
_LINESINDEX = {}
for i,l in enumerate(LINENAMES):
    _LINESINDEX[l] = i
    

# ============================ #
#                              #
#   Factory                    #
#                              #
# ============================ #

def linefit(filename,modelname="Mains", **kwargs):
    """ """
    return LinesFitter(filename, modelname="Mains",**kwargs)


# ============================ #
#                              #
#   Internal Functions         #
#                              #
# ============================ #
def gaussian_lines(lbda,x0,sigma,amplitudes=None):
    """ emission lines for the given lbda.
    (based on scipy's normal distribution)

    Parameters:
    -----------

    lbda: [array]
        (in Angstrom)
        Wavelength  where the lines will be measured.

    x0: [array-float]
        (in Angstrom)
        Location of the central emission lines

    sigma: [array-float]
       (in Angstrom)
        width of the emission lines  

    amplitudes: [array] -optional-
        amplitude of the different lines
        
    Returns
    -------
    array (flux)
    """
    
    gauss_array = stats.norm.pdf(np.asarray([lbda]*len(x0)).T,
                                 loc=x0, scale=sigma)
    if amplitudes is not None:
        return np.dot(gauss_array,amplitudes)
    
    return np.sum(gauss_array,axis=1)


# ============================ #
#                              #
#   Usable Priors              #
#                              #
# ============================ #
def lnprior_amplitudes(amplitudes):
    """ flat priors (in log) for amplitudes
    this returns 0 if the amplitudes are
    positives and -inf otherwise 
    """
    for a in amplitudes:
        if a<0: return -np.inf
    return 0

def lnprior_velocity(velocity, velocity_bounds):
    """ flat priors (in log) within the given boundaries
    this returns 0 if the velocity is within the boundaries
    and -inf otherwise.
    if velocity_bounds is None or both velocity_bounds[0] and velocity_bounds[1] are
    None, this will always returns 0
    """
    if velocity_bounds is None:
        return 0
    if velocity_bounds[0] is not None and velocity<velocity_bounds[0]:
        return -np.inf
    if velocity_bounds[1] is not None and velocity>velocity_bounds[1]:
        return -np.inf
    return 0

def lnprior_dispersion_flat(dispersion, dispersion_bounds):
    """ flat priors (in log) within the given boundaries
    this returns 0 if the dispersion is within the boundaries
    and -inf otherwise.
    if dispersion_bounds is None or both dispersion_bounds[0] and dispersion_bounds[1] are
    None, this will always returns 0
    """
    if dispersion_bounds is None:
        return 0
    if dispersion_bounds[0] is not None and dispersion<dispersion_bounds[0]:
        return -np.inf
    if dispersion_bounds[1] is not None and dispersion>dispersion_bounds[1]:
        return -np.inf
    return 0

def lnprior_dispersion(dispersion, loc, scale):
    """ Gaussian prior estimated from the good line measurement
    made based on flat priors.
    """
    return np.log(stats.norm.pdf(dispersion,loc=loc, scale=scale))
     

# ========================== #
#                            #
#     Fitter                 #
#                            #
# ========================== #
class LinesFitter( Spectrum, BaseFitter ):
    """ 
    """
    DERIVED_PROPERTIES = ["yfit","vfit","norm"]
    
    fit_chi2_acceptance = 0.1
    
    def __init__(self,filename=None,modelname = "Basic",
                 use_minuit=True,
                 **kwargs):
        """ Spectrum object upgraded to be a BaseFitter

        Parameters
        ----------

        Returns
        -------
        Void, init the object
        """
        
        # We did build before this avoids the build warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(LinesFitter,self).__init__(filename=filename,**kwargs)
            
        self.set_model(eval("LinesModel_%s()"%modelname))
        # --------------
        # - For the fit
        # loaded to later save time
        self._properties['use_minuit']   = use_minuit
        self._derived_properties['norm'] = np.abs(self.y.mean())
        self._derived_properties['yfit'] = self.y/self.norm
        if self.has_var:
            self._derived_properties['vfit'] = self.v/self.norm**2

    @_autogen_docstring_inheritance(BaseFitter.set_model,"BaseFitter.set_model")
    def set_model(self,*args,**kwargs):
        # doc from BaseFitter
        super(LinesFitter,self).set_model(*args,**kwargs)
        self.model.set_lbda(self.lbda)

        
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
        return -2 * self.model.get_loglikelihood(self.yfit,self.vfit,parameters)


    # ----------- #
    # - Plotter - #
    # ----------- #
    def show(self, savefile=None, show=True, ax=None,
             mcmc=False, nsample=100,
             variance_onzero=True, add_thumbnails=False,
             color="0.7", fitcolor= None, mcmccolor = None,
             **kwargs):
        """ display the data vs. the model

        Parameters:
        -----------
        savefile: [string/None]
            where to save the figure. If None, the plot won't be saved

        show: [bool]
            If the plot is not saved. It is shown except if this is False

        mcmc: [bool]
            If the mcmc has run, show examples there.

            
        variance_onzero: [bool]
            Display the variance around the y=0 axis instead of around
            the data-points

        color, fitcolor, mcmccolor: [matplotlib's colors]
            Colors of the data, the best fit, and the mcmc sampling (if any), respectively.
        
        """
        # --------------
        # - Plot init
        import matplotlib.pyplot as mpl 
        from astrobject.utils.mpladdon import specplot,figout
        self._plot = {}
        if ax is None:
            fig = mpl.figure(figsize=[13,5])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        elif "plot" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. ")
        else:
            fig = ax.figure
        
        # - Basics
        pl = ax.specplot(self.lbda,self.yfit,self.vfit,
                         color=color,err_onzero=variance_onzero,**kwargs)
        # - Prop of fit
        prop = dict(color = mpl.cm.Reds(0.7,1.) if fitcolor is None else fitcolor,
            lw=2, zorder=9)
            
        if mcmc:
            plmodel = self.display_mcmc_models(ax, nsample,
                                               basecolor = mcmccolor,
                                               **prop)
        else:
            plmodel = self.display_model(ax,self._fitparams, **prop) \
              if self.has_fit_run() else None

        # ---------
        # - Limits
        yextrem = np.percentile(self.yfit,[1,99])
        ax.set_ylim(yextrem[0] - (yextrem[1]-yextrem[0])*0.05,  yextrem[1] + (yextrem[1]-yextrem[0])*0.05)
        # ------
        self._plot['ax']     = ax
        self._plot['figure'] = fig
        self._plot['plot']   = [pl,plmodel]
        self._plot['prop']   = kwargs
        
        fig.figout(savefile=savefile,show=show,add_thumbnails=add_thumbnails)        
        return self._plot


    
    def show_mains(self,savefile=None, wavelength_window=100,
                   mcmc=False,**kwargs):
        """Zoom On Halpha NII + OII1,2 regions. See self.show for more details"""
        
        from astrobject.utils.mpladdon import figout
        from matplotlib.patches        import Rectangle
        import matplotlib.pyplot       as mpl

        if np.any([a not in self.model.freeparameters for a in ["OII1","HA"]]):
            raise NotImplementedError(" this method aim at displaying the OII1/OII2,HA and NII lines,"+\
                                      " at least one of them is not part of the model.")
        
        # -- Only Halpha and OII Known -- #
        
        guessed_Ha   = DICT_EMISSIONLINES["HA"]   * (1+self.model.velocity_guess/CLIGHT)
        guessed_OII1 = DICT_EMISSIONLINES["OII1"] * (1+self.model.velocity_guess/CLIGHT)
        flagHa  = (self.lbda> guessed_Ha-wavelength_window) &(self.lbda< guessed_Ha+wavelength_window)
        flagOII = (self.lbda> guessed_OII1-wavelength_window) &(self.lbda< guessed_OII1+wavelength_window)
        
    
        # ------------------- #
        # - Data Info       - #
        # ------------------- #            
        fout = self.get_fitvalues(mcmc=mcmc,nonsymerrors=True)
        
        # ------------------- #
        # - Setting         - #
        # ------------------- #
        fig   = mpl.figure(figsize=[10,6])
        axHa  = fig.add_axes([0.52,0.15,0.4,0.78])
        axOII = fig.add_axes([0.10,0.15, 0.4,0.78])
        detHa = fout["HA"]/fout["HA.err"][0]
        colorInfo = {
            "good":mpl.cm.Greens(0.8),
            "probably":mpl.cm.Blues(0.8),
            "maybe":mpl.cm.Oranges(0.8),
            "*Nope*":mpl.cm.Reds(0.8),
            "nothing":mpl.cm.Purples(0.8),
            }
        if detHa < 2:
            signal = "nothing"
        else:
            if detHa<3:
                signal = "maybe"
            elif detHa < 5:
                signal = "probably"
            else:
                signal = "good"
            
        # -------------------#
        #  Do The Plot       #
        # -------------------#
        self.show(savefile="_dont_show_",
                    ax=axHa, fitcolor=colorInfo[signal],
                    mcmc= mcmc, **kwargs)
        self.show(savefile="_dont_show_",
                    ax=axOII,fitcolor=colorInfo[signal],
                    mcmc= mcmc, **kwargs)
    
        axHa.legend([Rectangle((0, 0), 0, 0, alpha=0.0)], 
                ["Detection = "+"{:.1f} ".format(detHa)+signal], 
                handlelength=0, borderaxespad=0., loc="upper right",
                framealpha=0.7, labelspacing=2, fontsize="large"
            )
        axOII.legend([Rectangle((0, 0), 0, 0, alpha=0.0)], 
                ["OII = "+"{:.1f}".format((fout["OII1"]+fout["OII2"])/\
                            np.sqrt(fout["OII1.err"][0]**2+fout["OII2.err"][0]**2))], 
                handlelength=0, borderaxespad=0., loc="upper left", framealpha=0.7,
                labelspacing=2, fontsize="large"
            )
    
        # -------------------- #
        # - Shape it         - #
        # -------------------- #
        axHa.set_xlim(guessed_Ha-wavelength_window,
                      guessed_Ha+wavelength_window)
        axOII.set_xlim(guessed_OII1-wavelength_window,
                       guessed_OII1+wavelength_window)
        
        # - y bound
        ymin = np.min(np.concatenate([self.yfit[flagHa],self.yfit[flagOII]]))
        ymax = np.max(np.concatenate([self.yfit[flagHa],self.yfit[flagOII]]))
    
        axHa.set_ylim(ymin,ymax*1.1)
        axOII.set_ylim(ymin,ymax*1.1)
    
        for line in ["HA","NII_1","NII_2","OII1","OII2"]:
            axHa.axvline(DICT_EMISSIONLINES[line]*(1+fout["velocity"]/CLIGHT),
                         color="0.7",alpha=0.5)
            axOII.axvline(DICT_EMISSIONLINES[line]*(1+fout["velocity"]/CLIGHT),
                          color="0.7",alpha=0.5)
        # -------------------- #
        # - Fancy it         - #
        # -------------------- #
        from matplotlib.ticker      import NullFormatter
        axHa.yaxis.set_major_formatter(NullFormatter())
        axOII.set_ylabel(r"$\mathrm{Flux\ []}$",fontsize="x-large")
        fig.text(0.5,0.01,r"$\mathrm{Wavelength\ [\AA]}$",fontsize="x-large",
                 va="bottom",ha="center")
        
        fig.figout(savefile=savefile)
    
    # ------------------ #
    #   Plot Add on      #
    # ------------------ #
    def display_mcmc_models(self,ax, nsample, color=None,
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
        if not self.mcmc.has_ran():
            raise AttributeError("You need to run MCMC first.")
        
        if color is None:
            color     = mpl.cm.Blues(0.8,1)
        if basecolor is None:
            basecolor = mpl.cm.Blues(0.5,0.1)
        
        pl = [self.display_model(ax, param, color=basecolor,**kwargs)
              for param in self.mcmc.samples[np.random.randint(len(self.mcmc.samples), size=100)] ]
        
        self.display_model(ax, np.asarray(self.mcmc.derived_values).T[0],
                           color=color, **kwargs)
        
        return pl
    
    def display_model(self,ax,parameters,
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
        ymodel = self.get_model(parameters)
        ymodel[~self.model.lbdamask] = np.NaN
        return ax.plot(self.model.lbda,ymodel,
                       color=color,ls=ls,
                       **kwargs)
    # ====================== #
    # = Properties         = #
    # ====================== #
    @property
    def norm(self):
        return self._derived_properties["norm"]
    
    @property
    def yfit(self):
        # return self.y/self.norm
        return self._derived_properties["yfit"]

    @property
    def vfit(self):
        # return self.v/self.norm**2
        return self._derived_properties["vfit"]

    # ====================== #
    # = Properties         = #
    # ====================== #
    def _fit_readout_(self):
        #
        # Add Normalisation information
        # 
        super(LinesFitter,self)._fit_readout_()
        self.fitvalues["norm"]    = self.norm
        self.fitvalues["dof"]    = len(self.model.lbda[self.model.lbdamask])-self.model.nparam + \
          len(np.argwhere(np.asarray(self.paramfixed,dtype='bool')))
          
# ========================== #
#                            #
#     Models                 #
#                            #
# ========================== #
class LinesModel( BaseModel ):
    """ Mother of the Models

    # How to Create a new model:
    
    Models inherating fro LinesModels must have:
    
    * FREEPARAMETERS defined.
        To be able to use the default `parse_parameters()` method
        define `FREEPARAMETERS` as follow:
        `FREEPARAMETERS` = [LINE_1, LINE_2, ... LINE_N, `velocity`, `dispersion`]
        LINE_X could be known optical lines (see `LINENAMES`). If could also
        be 'NII' or 'OIII' ; if so the `NII_1/NII_2` and `OIII_1/OIII_2`
        ratio will be fixed.
        
    
    * parse_parameters(parameters) -optional-
       read the given parameters and returns
       amplitudes (for all the `LINENAMES`), `velocity`, `dispersion`.
       see the above `FREEPARAMETERS` definition to be able to use
       the default `parse_parameters()` method
       
    """
    PROPERTIES = ["lbda"]
    DERIVED_PROPERTIES = ["lbdamask"]


    # -- Generic values
    dispersion_guess      = 150
    dispersion_boundaries = [50,250]
    
    def __build__(self,*args,**kwargs):
        """ """
        super(LinesModel,self).__build__(*args,**kwargs)
        
        for l in self.freeparameters:
            if l in LINENAMES or l in DOUBLETS:
                self.__dict__['%s_boundaries'%l] = [0,None]
                self.__dict__['%s_guess'%l] = 1.
        
                
    # ========================= #
    # = Main Methods          = #  
    # ========================= #
    # ---------- #
    # - Getter - #
    # ---------- #
    def get_loglikelihood(self,flux,variance,parameters):
        """ return the likelihood of the model given the data

        Parameters
        ----------

        flux, variance: [array, array]
            data signal and associated variance (square of the errors)

        Return
        ------
        float (-0.5*chi2)
        """
        
        res = flux - self.get_model(parameters)
        
        return -0.5 * np.sum(res[self.lbdamask]**2/variance[self.lbdamask]) # Must be chi2 not per dof
        
    def get_model(self,parameters,**kwargs):
        """ return the flux of the modeled spectrum

        Parameters:
        -----------
        parameter: [array]
            array parsable by parse_parameters() for get_spectral_flux()

        Return
        ------
        array (flux to be compared to the data)
        """
        
        return self.get_spectral_flux(*self.parse_parameters(parameters),
                                      **kwargs)

    def get_spectral_flux(self,ampl,velocity_km_s, dispersion_km_s):
        """ creates the spectral flux model

        Parameters
        ----------

        ampl: [array of float]
            List of the amplitudes of all the Globally defined emission lines.

        velocity_km_s: [float]
            (in km/s)
            This defines the shift of the wavelength (1 + velocity_km_s / clight)
            Negative values mean blue-shifted while positive values mean redshifted.
        
        dispersion_km_s: [float]
            (in km/s)
            This width of the emission lines (sigma of the gaussian)
                         
        Returns
        -------
        flux [array of the same size as self.lbda]
        """

        # -- Location of the lines:
        x0 = LINEWAVES*(1. + velocity_km_s / CLIGHT)
        # -- dispersion
        dispersion = dispersion_km_s/CLIGHT * x0
        
        return gaussian_lines(self.lbda,
                              x0, dispersion,ampl)

    # ---------- #
    # - Setter - #
    # ---------- #
    def set_lbda(self, lbda, inwave=True, wavemin=100):
        """ wavelength used for the fit
        
        Parameter:
        ----------
        inwave: [bool] -optional-
            If the wavelength is given in log of wavelength (velocity step)
            and if this is true, the exponiential of lbda will be stored
            to have it in proper wavelength
        wavemin: [float] -optional-
            If inwave is True, the first wavelength of the given lbda will be
            compared to wavemin. If wavemin is bigger the code will assume the
            given lbda is in log scale.

        Return
        ------
        Void
        """
        
        self._properties['lbda'] = lbda if inwave is False or lbda.min()>wavemin \
          else np.exp(lbda)

    # This might not work if your FREEPARAMETERS are not Classic
    # [line1,line2..., velocity, dispersion]
    def parse_parameters(self,parameters):
        """  Basic parameter parse that assumes
        FREEPARAMETER = [list_of_wavelength, velocity, dispersion]
        
        Return
        -------
        [amplitude array], velocity, dispersion
        """
        ampl  = parameters[:-2]
        ampl_ = np.zeros(_NLINES)
        for l,v in zip(self.freeparameters[:-2],ampl):
            if l == "NII":
                ampl_[_LINESINDEX["NII_1"]] = v*0.34
                ampl_[_LINESINDEX["NII_2"]] = v
            elif l == "OIII":
                ampl_[_LINESINDEX["OIII_1"]] = v
                ampl_[_LINESINDEX["OIII_2"]] = v*2.98
            else:
                ampl_[_LINESINDEX[l]]        = v

        return ampl_, parameters[-2], parameters[-1]

    # ---------- #
    # - Priors - #
    # ---------- #
    def lnprior(self,parameters, loc=170, scale=15):
        """ Default Priors typical emission lines. """
        amplitudes, velocity, dispersion = self.parse_parameters(parameters)
        return lnprior_amplitudes(amplitudes) + \
          lnprior_velocity(velocity,self.velocity_boundaries) + \
          lnprior_dispersion(dispersion,loc=loc, scale=scale)
    
    # ========================= #
    # = Properties            = #  
    # ========================= #
    @property
    def lbda(self):
        """ the wavelength for which the model will be tested """
        return self._properties['lbda']

    @property
    def lbdamask(self, kept_width=80):
        """ mask for the used wavelengths """
        if self._derived_properties['lbdamask'] is None:
            if self.lbda is None:
                raise AttributeError("model's wavelength (self.lbda) has not been set")
            if self.param_input is None or \
              "velocity_guess" not in self.param_input.keys():
                raise AttributeError("Cannot make `lbdamask` without a velocity input")
            
            # - Setting the mask
            redshift = 1+ self.param_input['velocity_guess']/CLIGHT
            mask = np.zeros(len(self.lbda))
            for l in self.freeparameters:
                if l not in LINENAMES and l not in DOUBLETS:
                    continue
                lwaves = [DICT_EMISSIONLINES["%s_1"%l],DICT_EMISSIONLINES["%s_2"%l]] \
                  if l in DOUBLETS else [DICT_EMISSIONLINES[l]]
                for lwave in lwaves:
                    mask[(self.lbda > (lwave*redshift - kept_width))\
                       & (self.lbda < (lwave*redshift + kept_width))] = 1
                       
            self._derived_properties['lbdamask'] = np.asarray(mask,dtype=bool)
            
        # - the mask has been set
        return self._derived_properties['lbdamask']
    
    # ========================= #
    # = Internal              = #  
    # ========================= #
    
    
# -------------------------- #
#                            #
#  Actual Models             #
#                            #
# -------------------------- #

# --------------------- #
#  Normal lines Models  #
# --------------------- #
class LinesModel_All( LinesModel ):
    """ """
    FREEPARAMETERS = ["OII1","OII2","HN","HC","HE","HD","HG","HB",
                      "OIII","HA","NII","SII1","SII2",
                      "velocity","dispersion"]

class LinesModel_HaNII( LinesModel ):
    """ """
    FREEPARAMETERS = ["HA","NII",
                      "velocity","dispersion"]

        
class LinesModel_Mains( LinesModel ):
    """ """
    FREEPARAMETERS = ["HA","NII","OII1","OII2","SII1","SII2",
                       "velocity","dispersion"]
        


# --------------------- #
#  Trickier Models      #
# --------------------- #
class LinesModel_HaNIICont( LinesModel ):
    """ """
    FREEPARAMETERS = ["HA","NII","velocity","dispersion",
                      "cont","contslope"]
    contslope_guess = 0
    contslope_fixed = True
    def parse_parameters(self,parameters):
        """ read the parameter and parse them in a useful way
        
        Return
        -------
        [amplitude array], velocity, dispersion, cont, contslope
        """
        ampl, velocity, dispersion = super(LinesModel_HaNIICont,self).parse_parameters(parameters[:-2])
        cont, contspole = parameters[-2:]
        return ampl, velocity, dispersion,cont, contspole

    def get_spectral_flux(self,ampl,velocity_km_s, dispersion_km_s,
                          continuum, contslope):
        """creates the spectral flux model

        Parameters
        ----------

        ampl: [array of float]
            List of the amplitudes of all the Globally defined emission lines.

        velocity_km_s: [float]
            (in km/s)
            This defines the shift of the wavelength (1 + velocity_km_s / clight)
            Negative values mean blue-shifted while positive values mean redshifted.
        
        dispersion_km_s: [float]
            (in km/s)
            This width of the emission lines (sigma of the gaussian)

        continuum: [float]
            flat background under the lines (shared by all lines)

        contslope: [float]
            slope of the background. (b in total_cont = continuum + b*lbda)
            
            
        Returns
        -------
        flux [array of the same size as self.lbda]
        """
        elinesflux = super(LinesModel_HaNIICont,self).get_spectral_flux(ampl,velocity_km_s, dispersion_km_s)
        return elinesflux+continuum + self.lbda*contslope

    def lnprior(self,parameters, loc=170, scale=15):
        """ Default Priors typical emission lines. """
        amplitudes, velocity, dispersion, cont, contslope = self.parse_parameters(parameters)
        prior_cont,prior_contslope = 0,0
        return lnprior_amplitudes(amplitudes) + \
          lnprior_velocity(velocity,self.velocity_boundaries) + \
          lnprior_dispersion(dispersion,loc=loc, scale=scale) + prior_cont + prior_contslope
