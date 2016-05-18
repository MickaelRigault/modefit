#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Module to fit the main optical emission line """

import numpy       as np
from scipy.special import erf
from scipy import stats
import warnings
from astropy       import constants

# local dependency
from .baseobjects import BaseModel,BaseFitter

# astrobject dependencies
from astrobject.astrobject.spectroscopy import Spectrum
from astrobject.utils.decorators import _autogen_docstring_inheritance


__all__ = ["fit_spectrumlines"]


# ===================== #
# = Global Variables  = #
# ===================== #
CLIGHT=constants.c.to("km/s").value
# http://zuserver2.star.ucl.ac.uk/~msw/lines.html
DICT_EMISSIONLINES = {
    "OII1":3725.5,
    "OII2":3729,
    "HN":3835.4,
    "HC":3889.0,
    "HE":3970.,
    "HD":4101.8,
    "HG":4340.5,
    "HB":4861.3,
    "OIII_1":4958.9,
    "OIII_2":5007.,
    "HA":6562.8,
    "NII_1":6548.04,
    "NII_2":6583.45,
    "SII1":6716.3,
    "SII2":6730.7    
    }


# ========================== #
# = usage example          = #
# ========================== #
def fit_spectrumlines(spectrum_file,zguess=None,modelName="HaNIICont",
                      fit=True,fitprop={},**kwargs):
    """ fit a lineModel and the given Spectrum

    Parameters
    ---------
    spectrum_file: [fits file]
        A Spectrum fits file.
        Variance, if any, should be stored as the first
        hdu-table like SNIFS data are.

    zguess: [float or None]
        A guessed redshift for the given spectrum.
        If None is set, the default Model guess will be used

    modelName: [string]
        This is the name of the opticalLines Child Class
        See fitLinesModels library to have the list.
        The name of the class-Model shall be given without
        the prefixe 'OpticalLines'. For instance
        OpticalLines_Basic => modelName='Basic'
        Only a class-Model from this library are known.

    fitprop: [dictionary]
        Give options to the fit() function. This could be
        use_minuit or the _guess _fixed and _boundaries for
        the *freeParamters* of your model. e.g. HA_guess=...

    **kwargs goes to LinesFitter.__init__ which is a *Spectrum*

    Returns
    -------
    fitLine object, which is a child of the Spectrum object
    """
    lin = LinesFitter(spectrum_file,modelName=modelName,**kwargs)
    if fit:
        lin.fit(velocity_guess=zguess * CLIGHT, **fitprop)
    
    return lin


# ========================== #
# = Internal Functions     = #
# ========================== #
def get_lines_spectrum(x0,sigma,A,X,check_inputs=True,**kwargs):
    """ combine emission lines into a multiple gaussian array

    Parameters:
    -----------
    x0:    [float / float-array]
         = In Angstrom =
         Location of the central values of the emission line
         
    sigma  [float/ float-array]
         = In velocity (km/s) [or Angstrom if sigma_x0unit=True] =
         This is the width of the lines
         (could be a 1D array/float is all line share the same width)

    A  [float/ float-array]
         = Flux - unit =
         This is the Amplitude of the lines
         (could be a 1D array/float is all line share the same Amplitude)

    X      [array]
        = In Angstrom =
        Define the wavelength used to estimate the lines' pdf

    check_inputs: [bool] -optional-
        CAREFUL - set this to False enables to skip all the input tests
        use this if you know what you are doing in order to save time.
        
    **kwargs goes to `lines`. See this function options (e.g., velocity_step )
    
    Returns
    -------
    array (same size as X)
    """
    # ====================== #
    #  INPUT TEST            #
    # ====================== #
    if check_inputs:
        def make_me_iterable(a):
            return a if "__iter__" in dir(a) else [a]
        
        x0, sigma, A = \
          make_me_iterable(x0),make_me_iterable(sigma),make_me_iterable(A)

        # -- Is Sigma Input Ok ?
        if len(sigma) == 1:
            sigma = sigma*len(x0)
        elif len(sigma) != len(x0):
            raise ValueError("sigma and x0 must have the same lengths except if sigma is a unique value (%d vs. %d)"%(len(sigma),len(x0)))
        # -- Is Amplitude Input Ok ?
        if len(A) == 1:
            A = A[0]*len(x0)
        elif len(A) != len(x0):
            raise ValueError("A and x0 must have the same lengths except if A is a unique value (%d vs. %d)"%(len(A),len(x0)))

    # ====================== #
    #  The Spectra           #
    # ====================== #
    # -- Things looks great, let's build the spectrum
    return np.sum([ A* line(x0,sigma,X,**kwargs)
                    for x0,sigma,A in zip(x0,sigma,A)],
                    axis=0)

def line(x0,sigma,X,normalized=False,
         velocity_step=False,sigma_x0unit=False):
    """ Get a gaussian emission line (based on erf)

    Parameters
    ----------
    x0:    [float]
        = In Angstrom =
        Location of the central value of the emission line
         
    sigma: [float]
        = In velocity (km/s) [or Angstrom if sigma_x0unit=True] =
        This is the width of the line

    X:     [array]
        = In Angstrom =
        Define the wavelength used to estimate the line pdf

    normalized: [bool]
        Set to False to skip the normalisation.
         
    velocity_step:  [bool] = Set True in ULySS Fit case =
        Set True if the step between the X points are linear in ln(X).
        (If this is a regular spectrum, the step shall be constant in wavelength)
          
    sigma_x0unit: [bool]
        Set True if the sigma parameter is given in Angstrom and not in km/s

        
    Returns
    -------
    array (same size as X)    
    """
    
    # dispersion in pixel space and relativ position
    npix        = len(X)# npix same if log or not
    sigmax0unit = sigma if sigma_x0unit else sigma/CLIGHT * x0
    
    # stepreg --> x0*step
    if velocity_step is True:
        lamrangelog = np.log(X)
        start = lamrangelog.min()
        end   = lamrangelog.max()
        
        step  = (end-start)/(npix-1)
        pos = (np.log(x0) - start) / step # take it in pixel 
        norm_erf   = 1/(x0*step)
        sigma_pix  = sigmax0unit / (x0*step)    
    else:
        ## regular size
        startreg = X.min()
        endreg   = X.max()
        stepreg  = (endreg -startreg )/(npix-1)
    
        pos = (x0 - startreg) / stepreg # take it in pixel 
        norm_erf   = 1/stepreg
        sigma_pix  = sigmax0unit / stepreg
        
    xborder = np.arange(npix+1) - 0.5
    # gaussian exposant
    y = (xborder -pos) / sigma_pix
    # analitics integration
    cumul = 0.5*(1+erf(y/np.sqrt(2)))
    # numerics derivation
    data = (np.roll(cumul,-1)[0:npix]-cumul[0:npix])*norm_erf
    
    # default : erf method is normalized
    if normalized is False:
        # remove normalisation
        norm = 1./(sigmax0unit*np.sqrt(2*np.pi))
        data /=norm

    return data

# ============================ #
#                              #
#  Usable Priors             = #
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

def lnprior_dispersion(dispersion, loc=170, scale=15):
    """ Gaussian prior estimated from the good line measurement
    made based on flat priors.
    """
    return np.log(stats.norm(loc=loc, scale=scale).pdf(dispersion))
     


########################################################
#                                                      #
#      LINE FITTER CLASS                               #
#                                                      #
########################################################

class LinesFitter( Spectrum, BaseFitter ):
    """ object to enable the fit the line using the
    model in lineModels. The fitting technique is based on
    the Virtual ScipyMinuit Classes in Fitter_library.
    """
    
    fit_chi2_acceptance = 0.1
    # ====================== #
    # = Hack of Spectrum   = #
    # ====================== #    
    @_autogen_docstring_inheritance(Spectrum.__init__,"Spectrum.__init__")
    def __init__(self,filename=None,modelName = "Basic",
                 use_minuit=True,
                 **kwargs):
        #
        # Add the model parameters
        #
        
        self.set_model(eval("OpticalLines_%s()"%modelName))
        
        # We did build before this avoids the build warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(LinesFitter,self).__init__(filename=filename,**kwargs)

        # -- For the fit
        self.use_minuit = use_minuit
        self.norm = np.abs(self.y.mean())
        # - fit in flux space 
        self.yfit = self.y/self.norm
        
        if self.has_var:
            self.vfit = self.v/self.norm**2
        
    @_autogen_docstring_inheritance(Spectrum.set_lbda,"Spectrum.set_lbda")
    def set_lbda(self,x):
        #
        # Add on: Create the model lbda together
        #
        super(LinesFitter,self).set_lbda(x)
        self.model.set_lbda(x)

        
    @_autogen_docstring_inheritance(Spectrum._load_lbda_,"Spectrum._load_lbda_")
    def _load_lbda_(self):
        #
        # Add on: Create the model lbda together
        #
        super(LinesFitter,self)._load_lbda_()
        if self.has_velocity_step():
            self.model.set_lbda(np.log(self.lbda))
        else:
            self.model.set_lbda(self.lbda)
        
    @_autogen_docstring_inheritance(Spectrum.copy,"Spectrum.copy")
    def copy(self,*args,**kwargs):
        #
        # Add on: Create the model lbda together
        #
        super(LinesFitter,self).copy(*args,**kwargs)
        if self.has_velocity_step():
            self.model.set_lbda(np.log(self.lbda))
        else:
            self.model.set_lbda(self.lbda)
            
    # =========================== #
    # = Hack of VirtualFitter   = #
    # =========================== #
    @_autogen_docstring_inheritance(BaseFitter.fit,"BaseFitter.fit")
    def fit(self,*args,**kwargs):
        #
        # - Fitting acceptance
        #
        super(LinesFitter,self).fit(*args,**kwargs)
        if self.fit_chi2_acceptance is not None and \
          np.abs(self.fitvalues["chi2dof"] - 1) > self.fit_chi2_acceptance:
            self.v    *= self.fitvalues["chi2dof"]
            self.vfit *= self.fitvalues["chi2dof"]
            k_newguess = {}
            for name in self.model.freeparameters:
                k_newguess["%s_guess"%name] = np.abs(self.fitvalues[name] + np.random.rand(1)*10)
            self.fit(**k_newguess)
        

    
    @_autogen_docstring_inheritance(BaseFitter.setup_guesses,"BaseFitter.setup_guesses")
    def setup_guesses(self,**kwargs):
        #
        # Add default velocity boundaries
        #
        super(LinesFitter,self).setup_guesses(**kwargs)
        if "velocity" in self.model.freeparameters and self.model.velocity_boundaries == [None,None]:
            self.model.velocity_boundaries = [self.model.velocity_guess - self.model.VELOCITY_BOUNDWINDOW,
                                              self.model.velocity_guess + self.model.VELOCITY_BOUNDWINDOW]
            self.param_input["velocity_boundaries"] = self.model.velocity_boundaries
            
    @_autogen_docstring_inheritance(BaseFitter._fit_readout_,"BaseFitter._fit_readout_")
    def _fit_readout_(self):
        #
        # Add Normalisation information
        #
        # - Update the model 
        self.model.update(*self.model.parse_parameters(self._fitparams))
        #
        super(LinesFitter,self)._fit_readout_()
        self.dof       = len(self.model.y[self.model.lbda_mask])-self.model.nparam + \
          len(np.argwhere(np.asarray(self.paramfixed,dtype='bool')))
        if "object" in dir(self) and \
          self.object is not None and self.object!="unknown":
            self.fitvalues["object"]  = self.object
            
        self.fitvalues["norm"]    = self.norm
        self.fitvalues["chi2dof"] = self.fitvalues["chi2"] / self.dof
        
    # ========================= #
    # = Fitting Particularity = #
    # ========================= #
    def get_residual(self,parameters,**kwargs):
        """
        **kwargs goes to updateModel
        """
        return self.yfit - self.model.get_model(parameters,**kwargs)

    
    def get_modelchi2(self,parameters,**kwargs):
        """ chi2 model that has to be given to the model

        if a variance is defined, a Chi2/dof is used (least square otherwise)
        
        The wavelength mask of the model will be applied, such that
        the chi2 is measured only for the modeled spectral regions.
        => see opticalLines.generate_wavelength_mask()

        Parameters
        ----------
        parameters: [array]
            [ampl, velocity_km_s, dispersion]:
              * ampl = Amplitude array
              * velocity_km_s = float [in km/s]
              * dispersion = float [in km/s]
        Returns
        -------
        float (chi2)
        """
        res = self.get_residual(parameters,**kwargs)
        if self.has_var:
            chi2 =  np.sum(res[self.model.lbda_mask]**2/self.vfit[self.model.lbda_mask]) # Must be chi2 not per dof
        else:
            chi2 = np.sum(res[self.model.lbda_mask]**2)

        if chi2 != chi2:
            print "Chi2 is Nan"
            print "  parameters : "," ,".join(np.asarray(parameters,dtype="str"))
            raise ValueError("Failure in Chi2 minimization")

        return chi2

    # =============================== #
    # =  Spectrum Adaptation        = #
    # =============================== #
    def show_mcmcfit(self, nsamples = 100,
                     savefile=None, show=True,
                     variance_onzero=False,ax=None):
        """ Some typical mcmc examples overplotting the data

        Parameters
        ----------

        savefile: [string] -optional-
            where to save the plot if you wish so (set None if not)

        show: [bool] -optional-

        Returns
        -------
        dictionnary (plot output)
        """
        
        if not self.has_mcmc() or not self.mcmc.has_run():
            raise AttributeError("Run MCMC first.")

        if nsamples> self.mcmc.nsamples:
            raise ValueError("More samples requested than mcmc samples (%d vs. %d)"%(nsamples,self.mcmc.nsamples))
        
        # --------------
        # - Plot init
        # --------------
        import matplotlib.pyplot as mpl 
        from astrobject.utils.mpladdon import specplot,figout
        self._plot = {}
        if ax is None:
            fig = mpl.figure(figsize=[8,5])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        elif "plot" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. ")
        else:
            fig = ax.figure

        # -----------
        # - Data
        # -----------
        pl = ax.specplot(self.lbda,self.yfit,self.vfit,
                         color="0.4",err_onzero=variance_onzero)
        # ----------------
        # - MCMC examples
        # ----------------
        #self.mcmc.samples[np.random.randint(self.mcmc.nsamples, size=nsamples)]
        #[ax.specplot(self.lbda,self.yfit,self.vfit,
        #                 color="0.4",err_onzero=variance_onzero)
        
        
    def show(self,savefile=None,show=True,
             variance_onzero=False,ax=None,
             modelprop={},add_thumbnails=False,
             **kwargs):
        """
        = ONLY BASICS NEEDS A MAJOR UPDATE =
        """
        if "yfit" not in dir(self):
            super(LinesFitter,self).show(savefile=savefile,**kwargs)
            return

        # --------------
        # - Plot init
        import matplotlib.pyplot as mpl 
        from astrobject.utils.mpladdon import specplot,figout
        self._plot = {}
        if ax is None:
            fig = mpl.figure(figsize=[8,5])
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
        elif "plot" not in dir(ax):
            raise TypeError("The given 'ax' most likely is not a matplotlib axes. ")
        else:
            fig = ax.figure
        
        
        # - Basics
        pl = ax.specplot(self.lbda,self.yfit,self.vfit,
                         color="0.4",err_onzero=variance_onzero)

        plmodel = ax.plot(self.model.lbda,self.model.y,"r-", alpha=0.2)
        # -- Actual Fit
        xfit = self.model.lbda.copy()
        yfit = self.model.y.copy()

        xfit[-self.model.lbda_mask] = np.NaN
        yfit[-self.model.lbda_mask] = np.NaN

        color=modelprop.pop("color","r")
        ax.plot(xfit,yfit,
                color=color,zorder=8,
                **modelprop)

        ymin = np.nanmin(self.yfit)
        xmin = np.nanmin(self.lbda)
        ymax = np.nanmax(self.yfit)
        if ymax < abs(ymin):
            ymin = -ymax
        ax.set_ylim((ymin-abs(ymin*0.10), np.nanmax(self.yfit)*1.1))
        ax.set_xlim((xmin-abs(xmin*0.01), np.nanmax(self.lbda)*1.01))

        self._plot['ax']     = ax
        self._plot['figure'] = fig
        self._plot['plot']   = [pl,plmodel]
        self._plot['prop']   = kwargs
        
        fig.figout(savefile=savefile,show=show,add_thumbnails=add_thumbnails)        
        return self._plot
        
    def show_HaNII(self,savefile=None,wavelength_window=100,**kwargs):
        """Zoom On Halpha NII regions. See self.show for more details"""

        from astrobject.utils.mpladdon import figout
        
        self.show(savefile="_dont_show_",**kwargs)
        zguess    = self.model.velocity_guess / CLIGHT
        zmeasured = self.minuit.values["velocity"] / CLIGHT
        guessed_Halpha  = self.model.linewave[self.model._indexHalpha] * (1+zguess)
        measured_Halpha = self.model.linewave[self.model._indexHalpha] * (1+zmeasured)
        
        self._plot["ax"].axvline(guessed_Halpha,color="k",alpha=0.2)
        self._plot["ax"].ax.axvline(measured_Halpha,color=self._modelColor,alpha=0.2)
        self._plot["ax"].ax.set_xlim(guessed_Halpha-wavelength_window,
                              guessed_Halpha+wavelength_window)
        self._plot["fig"].figout(savefile)

    def show_mains(self,savefile=None,wavelength_window=100,**kwargs):
        """Zoom On Halpha NII + OII1,2 regions. See self.show for more details"""

        from astrobject.utils.mpladdon import figout
        from matplotlib.patches import Rectangle
        
        variance_onzero = kwargs.pop("variance_onzero",True)

        # -- Only Halpha Known -- #
        if "OII1" not in self.model.freeparameters:
            self.show_HaNII(savefile=savefile,variance_onzero=variance_onzero,
                            **kwargs)
            return
        
        import matplotlib.pyplot as mpl
        
        # -- Only Halpha and OII Known -- #
        
        guessed_Ha  = self.model.linewave[self.model._indexHalpha] * (1+self.model.velocity_guess/CLIGHT)
        guessed_OII1     = self.model.linewave[self.model._indexOII1]  * (1+self.model.velocity_guess/CLIGHT)
        flagHa = (self.lbda> guessed_Ha-wavelength_window) &(self.lbda< guessed_Ha+wavelength_window)
        flagOII = (self.lbda> guessed_OII1-wavelength_window) &(self.lbda< guessed_OII1+wavelength_window)
        
        # ------------------- #
        # - Setting         - #
        # ------------------- #
        fig   = mpl.figure(figsize=[10,6])
        axHa  = fig.add_axes([0.52,0.15,0.4,0.78])
        axOII = fig.add_axes([0.10,0.15, 0.4,0.78])
        detHa = self.fitvalues["HA"]/self.fitvalues["HA.err"]
        colorInfo = {
            "good":mpl.cm.Greens(0.8),
            "probably":mpl.cm.Blues(0.8),
            "maybe":mpl.cm.Oranges(0.8),
            "*Nope*":mpl.cm.Reds(0.8),
            "nothing":mpl.cm.Purples(0.8),
            }
        if   detHa < 2:
            signal = "nothing"
        else:
            if self.is_fit_good() is False:
                signal = "*Nope*"
            elif detHa<3:
                signal = "maybe"
            elif detHa < 5:
                signal = "probably"
            else:
                signal = "good"
            
        # -------------------- #
        # - Do The Plot      - #
        # -------------------- #
        modelprop = dict(color=colorInfo[signal],lw=2)
        self.show(savefile="_dont_show_",variance_onzero=variance_onzero,
                  ax=axHa,modelprop=modelprop,**kwargs)
        modelprop = dict(color=colorInfo[signal],lw=2)
        self.show(savefile="_dont_show_",variance_onzero=variance_onzero,
                  ax=axOII,modelprop=modelprop,**kwargs)

        axHa.legend([Rectangle((0, 0), 0, 0, alpha=0.0)], 
                ["Detection = "+"{:.1f} - ".format(detHa)+signal], 
                handlelength=0, borderaxespad=0., loc="upper right", framealpha=0.7, labelspacing=2, fontsize="large"
            )

        axOII.legend([Rectangle((0, 0), 0, 0, alpha=0.0)], 
                ["Detection = "+"{:.1f}".format((self.fitvalues["OII1"]+self.fitvalues["OII2"])/\
                                                 np.sqrt(self.fitvalues["OII1.err"]**2+self.fitvalues["OII2.err"]**2))], 
                handlelength=0, borderaxespad=0., loc="upper left", framealpha=0.7, labelspacing=2, fontsize="large"
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
            linindex = np.argwhere(self.model.LINENAMES == line)[0][0]
            axHa.axvline(self.model.linewave[linindex]*(1+self.fitvalues["velocity"]/CLIGHT),
                                 color="0.7",alpha=0.5)
            axOII.axvline(self.model.linewave[linindex]*(1+self.fitvalues["velocity"]/CLIGHT),
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



########################################################
#                                                      #
#      LINE MODEL CLASSES                              #
#                                                      #
########################################################

# ================== #
#  Mother Class    = #
# ================== #
class _OpticalLines_( Spectrum ):
    """ """

    PROPERTIES = []
    SIDE_PROPERTIES = []
    DERIVED_PROPERTIES = ["lbda_mask","nampl"]
    # ---------
    # - Add on
    LINENAMES   = np.asarray(["OII1", "OII2","HN","HC","HE","HD","HG","HB",
                    "OIII_1","OIII_2","HA","NII_1","NII_2",
                    "SII1","SII2"])
    COMBINEDLINES = ["NII","OIII"]
    # -- Default velocity boundaries
    VELOCITY_BOUNDWINDOW = 2000
    # -- Analyzed Wavelength Area
    KEPT_WAVEWIDTH   = 80 # in Angstrom

    
    # -- Generic values
    dispersion_guess      = 150
    dispersion_boundaries = [50,250]

    # ========================= #
    # = Initialization        = #  
    # ========================= #
    def __build__(self,*args,**kwargs):
        """ """
        super(_OpticalLines_,self).__build__(*args,**kwargs)
        
        for name in self.freeparameters:
            if name in self.LINENAMES:
                self.__dict__['%s_boundaries'%name] = [0,None]
        

    # ========================= #
    # = Main Methods          = #  
    # ========================= #
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
    
    def parse_parameters(self,parameters):
        """ Generique function that works only if the model is made
        of emission line amplitudes and velocity+dispersion.
        This converts the given parameter into a readable
        self.update input.
        Return
        -------
        [amplitude array], velocity, dispersion
        """
        return parameters[:-2],parameters[-2],parameters[-1]

            
    def update(self,ampl,velocity_km_s,
               dispersion,**kwargs):
        """ create the spectrum based on the given parameters

        Parameters
        ---------- 
        ampl: [array]
            This is a list of parameter (e.g. amplitudes) that
            is given to the read_amplitudes function.
            This, in turn, defines the individual emission lines
            amplitudes. (see read_amplitudes)

        velocity_km_s: [float] (in km/s)
            This defines the shift of the wavelength (1 + velocity_km_s / clight)
            Negative values mean blue-shifted while positive values mean redshifted.
        
        dispersion: [float]
            (km/s or pixel if dispersion_inPixel is True)
            This width of the emission lines (sigma of the gaussian)

        **kwargs goes to get_spectral_flux

        Returns
        -------
        Void ; (defines self.lbda, self.y)
        
        """
        self.y =  self.get_spectral_flux(ampl,velocity_km_s,
                                        dispersion,**kwargs)
                                                                              
    # update and this are splited to allow easy inheritance.
    def get_spectral_flux(self,ampl,velocity_km_s,dispersion,
                          dispersion_inPixel=False):
        """ creates the spectral flux model

        Parameters
        ----------
        
        ampl: [array of float]
            This is a list of parameter (e.g. amplitudes) that is given
            to the read_amplitudes function. This, in turn, defines the
            individual emission lines amplitudes.
            (see read_amplitudes)
            
        velocity_km_s: [float]
            (in km/s)
            This defines the shift of the wavelength (1 + velocity_km_s / clight)
            Negative values mean blue-shifted while positive values mean redshifted.
        
        dispersion: [float]
            (km/s or pixel if dispersion_inPixel is True)
            This width of the emission lines (sigma of the gaussian)
                 
        dispersion_inPixel: [bool]
            = default is False =
            Set to True if the dispersion is given in pixel rather than if km/s
        
        Returns
        -------
        flux [array of the same size as self.lbda]
        """
        return  get_lines_spectrum(self.linewave*(1. + velocity_km_s / CLIGHT),
                                   [dispersion]*len(self.linewave),
                                   self.read_amplitudes(ampl),
                                   self.lbda,normalized=True,
                                   sigma_x0unit = dispersion_inPixel,
                                   velocity_step = self.has_velocity_step(),
                                   check_inputs=False)
    
    # ----------------
    # - Bayesian Touch
    def lnprior(self,parameters):
        """
        sum the parameters priors.
        prior function from the module
        lnprior_amplitudes, lnprior_velocity, lnprior_dispersion
        """
        amplitudes,velocity,dispersion = self.parse_parameters(parameters)
        
        return lnprior_amplitudes(amplitudes) + lnprior_velocity(velocity,self.velocity_boundaries)+ \
          lnprior_dispersion(dispersion)

    # ================= #
    # = Properties    = #
    # ================= #
    @property
    def _nopticallines(self):
        return len(self.LINENAMES)

    @property
    def linewave(self):
        """ array of the emmission line waves """
        return np.asarray([DICT_EMISSIONLINES[k] for k in
                    self.LINENAMES])

    @property
    def usedlines(self):
        """ sorted list of the used lines """
        return np.concatenate([[l] if l not in self.COMBINEDLINES else [l+"_1",l+"_2"]  for
                               l in self.freeparameters if l in self.LINENAMES
                               or l in self.COMBINEDLINES])
    @property
    def nampl(self):
        """ number of amplitude parameters. """
        # this is stored because if is call at each fitting iteration.
        if self._derived_properties["nampl"] is None:
            self._derived_properties["nampl"] = len([l for l in self.freeparameters
                                  if l in self.COMBINEDLINES or l in self.LINENAMES])
        return self._derived_properties["nampl"]
    
    # --------------
    # - Da Mask
    @property
    def lbda_mask(self):
        """ mask for the wavelength array.

        All the wavelength will be masked except the one corresponding
        to the emission-line listed in self.usedlines. A acceptance of
        self.KEPT_WAVEWIDTH won't be mask around this lines.

        This mask is generated the first time is it called.
        """
        if self._derived_properties['lbda_mask'] is None:
            self._derived_properties['lbda_mask'] = np.asarray(np.zeros(len(self.lbda)),dtype = "bool")
            if "velocity_guess" not in dir(self):
                warnings.warns("No velocity guess given, 0 is set while loaded the lbda_mask")
                velocity_guess = 0
            else:
                velocity_guess = self.velocity_guess
                
            redshift = ( 1. + velocity_guess / CLIGHT )
            for line in self.usedlines:
                index = np.argwhere(self.LINENAMES == line)[0][0]
                flagkept = (self.lbda > (self.linewave[index]*redshift - self.KEPT_WAVEWIDTH))\
                & ((self.lbda < self.linewave[index]*redshift + self.KEPT_WAVEWIDTH))
                self._derived_properties['lbda_mask'][flagkept] = True
                
        return self._derived_properties['lbda_mask']


# =================================== #
# =  Models                         = #
# =================================== #
class OpticalLines_Basic( _OpticalLines_,BaseModel ):
    """
    """
    FREEPARAMETERS = ["OII1","OII2","HN","HC","HE","HD","HG","HB",
                       "OIII","HA","NII","SII1","SII2",
                       "velocity","dispersion"]
    
    _indexHalpha  = np.argwhere(_OpticalLines_.LINENAMES == "HA")[0][0]
    _indexNIIR    = np.argwhere(_OpticalLines_.LINENAMES == "NII_2")[0][0]
    _indexNIIB    = np.argwhere(_OpticalLines_.LINENAMES == "NII_1")[0][0]
    _indexOII1    = np.argwhere(_OpticalLines_.LINENAMES == "OII1")[0][0]
    _indexOII2    = np.argwhere(_OpticalLines_.LINENAMES == "OII2")[0][0]


    # ==================== #
    # =  Model           = #
    # ==================== #
    def read_amplitudes(self,ampl,
                        global_ampl=1.):
        """ reads the Amplitude and return a effective_amplitude array

        Parameters
        ----------
        ampl: [array of float]
            = amplitudes of the 13 lines =
            Give a list of amplitude for all the line, from the bluest to
            the reddest ; see self.LINENAMES
            This Model:
               `ampl` does not contain OIII-red nor NII-Blue, which amplitude
               are fixed-ratio amplitude doublette
                
        global_ampl: [float]
            = Multiplicative parameter =
            All the amplitude will be multiplied by this.

        Returns
        -------
        array of float ; the amplitudes of the emission lines.
        (See self.LINENAMES)
        """
        if len(ampl) != self.nampl:
            print ampl
            raise ValueError("ampl must have exactly %d entry. %d given"%(self.nampl,len(ampl)))
        
        ampl_ = ampl.copy()
        ampl_ = np.insert(ampl_,9,ampl_[8]*2.98) # add associated OIII (add 5007)
        ampl_ = np.insert(ampl_,11,ampl_[11]*0.34) # add associated NII  (add 6548)

        return np.asarray(ampl_)*global_ampl

    
    # -------------------- #
    # - Allow Minuit Fit - #
    # -------------------- #
    def _minuit_chi2changed_(self,OII1,OII2,HN,HC,HE,HD,HG,HB,
                       OIII,HA,NII,SII1,SII2,
                       velocity,dispersion):
        """ Stupid function such that minuit and scipy can be used similarly
        (Give here all the parameter using the freeparameters order)
        """
        
        parameter = np.asarray([OII1, OII2,
                               HN, HC, HE, HD, HG, HB,
                               OIII, HA, NII, SII1, SII2,
                               velocity,dispersion])
            
        return self.get_chi2(parameter)

# ------------------------------- #
# --   Halpha NII  Model       -- #
# ------------------------------- #
class OpticalLines_HaNII( _OpticalLines_,BaseModel  ):
    """
    """
    FREEPARAMETERS = ["HA","NII", "velocity","dispersion"]
    
    _indexHalpha  = np.argwhere(_OpticalLines_.LINENAMES == "HA")[0][0]
    _indexNIIR    = np.argwhere(_OpticalLines_.LINENAMES == "NII_2")[0][0]
    _indexNIIB    = np.argwhere(_OpticalLines_.LINENAMES == "NII_1")[0][0]

    # ==================== #
    # =  Model           = #
    # ==================== #
    def read_amplitudes(self,ampl,
                        global_ampl=1.):
        """ reads the Amplitude and return a effective_amplitude array =

        Parameters
        ----------
        ampl: [array of float]
            = amplitudes of the lines  =
            Give a list of amplitude for all the Halpha and NII lines
            This Model:
                `ampl` Only Contains 1 NII line amplitude since the 2
                NII line have a fixed ration.
                The Other optical lines are set to 0.
                
        global_ampl: [float]
            = Multiplicative parameter =
            All the amplitude will be multiplied by this.

        Returns
        --------
        array of float ; the amplitudes of the emission lines.
        (See self.LINENAMES)
        """
        if len(ampl) != self.nampl:
            raise ValueError("ampl must have exactly %d entry. %d given"%(self.nampl,len(ampl)))

        ampl_ = np.zeros(self._nopticallines)
        ampl_[self._indexHalpha] = ampl[0]
        ampl_[self._indexNIIR]   = ampl[1]
        ampl_[self._indexNIIB]   = ampl[1]*0.34

        return np.asarray(ampl_)*global_ampl
        
    # -------------------- #
    # - Allow Minuit Fit - #
    # -------------------- #
    
    def _minuit_chi2changed_(self,HA,NII,
                       velocity,dispersion):
        """ Stupid function such that minuit and scipy can be used similarly
        (Give here all the parameter using the freeparameters order)
        """
        parameter = np.asarray([HA,NII,
                    velocity,dispersion])
            
        return self.get_chi2(parameter)

# ------------------------------- #
# --   Halpha NII + Cont       -- #
# ------------------------------- #
class OpticalLines_HaNIICont( OpticalLines_HaNII ):
    """
    """
    FREEPARAMETERS = ["HA","NII",
                      "cont0","contSlope",
                      "velocity","dispersion"]
    
    
    contSlope_guess = 0
    contSlope_fixed = True
    # -------------------- #
    # - Hack of HaNII    - #
    # -------------------- #
    def get_spectral_flux(self,*arg,**kwargs):
        """
        """
        flux = super(OpticalLines_HaNIICont,self).get_spectral_flux(*arg,**kwargs)
        cont = self.cont0 + self.contSlope*self.lbda
        return flux + cont
    
    def parse_parameters(self,parameters):
        """ Generique function that works only if the model is made
        of emission line amplitudes and velocity+dispersion
        """
        self.cont0,self.contSlope = parameters[2:-2]
        return parameters[:2],parameters[-2],parameters[-1]
    
    # -------------------- #
    # - Allow Minuit Fit - #
    # -------------------- #
    def _minuit_chi2changed_(self,HA,NII,cont0,contSlope,
                       velocity,dispersion):
        """ Stupid function such that minuit and scipy can be used similarly =
        (Give here all the parameter using the freeparameters order)
        """
        parameter = np.asarray([HA,NII,cont0,contSlope,
                               velocity,dispersion])
            
        return self.get_chi2(parameter)

# ------------------------------- #
# --  Halpha NII  Model Balmer -- #
# ------------------------------- #
class OpticalLines_BalmerSerie(  _OpticalLines_,BaseModel ):
    """
    """
    FREEPARAMETERS = ["HA","NII",
                      "ebmv","Rv",
                      "velocity","dispersion"]
    
    
    _indexHalpha  = np.argwhere(_OpticalLines_.LINENAMES == "HA")[0][0]
    _indexHbeta   = np.argwhere(_OpticalLines_.LINENAMES == "HB")[0][0]
    _indexHgamma  = np.argwhere(_OpticalLines_.LINENAMES == "HG")[0][0]
    _indexHdelta  = np.argwhere(_OpticalLines_.LINENAMES == "HD")[0][0]
    _indexHepsilon= np.argwhere(_OpticalLines_.LINENAMES == "HE")[0][0]
    _indexNIIR    = np.argwhere(_OpticalLines_.LINENAMES == "NII_2")[0][0]
    _indexNIIB    = np.argwhere(_OpticalLines_.LINENAMES == "NII_1")[0][0]

    # -- Analyzed Wavelength Area
    @property
    def usedlines(self):
        """ special line case """
        return ["HA","HB","HG","HD","HE", "NII_1","NII_2"]
    
    # using http://cdsads.u-strasbg.fr/abs/1971Ap%26SS..10..383G
    #   - averaged 1 and 2 to fit the known usual ratio Ralpha/beta = 2.8
    Ralphabeta    = 3.0  # could be up to 3.1
    Ralphagamma   = 5.93
    Ralphadelta   = 9.8  # quite unsure
    Ralphaepsilon = 15.1 # quite unsure
    Ralphazeta    = 21.7 # quite unsure
    # -- Some constrain
    ebmv_guess = 0.2
    ebmv_boundaries = [-0.1,0.6]

    Rv_guess      = 3.1
    Rv_boundaries = [1.5,5]
    Rv_fixed      = False
    
    # ==================== #
    # =  Model           = #
    # ==================== #
    def read_amplitudes(self,ampl,
                        global_ampl=1.):
        """ reads the Amplitude and return a effective_amplitude array =

        Parameters
        ----------
        ampl: [array of float]
            = amplitudes of the 2 lines  =
            Give a list of amplitude for all the Halpha and NII lines
            This Model:
               `ampl` Only Contains 1 NII line amplitude since the 2
               NII line have a fixed ration.
               The Other optical lines are set to 0.
                
        global_ampl: [float]
            = Multiplicative parameter =
            All the amplitude will be multiplied by this.

        Returns
        -------
        array of float ; the amplitudes of the emission lines.
        (See self.LINENAMES)
        """
        if len(ampl) != self.nampl:
            raise ValueError("ampl must have exactly %d entry. %d given"%(self.nampl,len(ampl)))
        self.load_extinction() # this is skiped is already loaded
        
        ampl_ = np.zeros(self._nopticallines)
        
        ampl_[self._indexHalpha] = ampl[0] * self.extinctionFactor(self.linewave[self._indexHalpha],
                                                                   self.ebmv,self.Rv)
        ampl_[self._indexNIIR]   = ampl[1] * self.extinctionFactor(self.linewave[self._indexNIIR],
                                                                   self.ebmv,self.Rv)
        ampl_[self._indexNIIB]   = ampl[1]*0.34 * self.extinctionFactor(self.linewave[self._indexNIIB],
                                                                        self.ebmv,self.Rv)
          
        # - rest of the Balmer Serie
        ampl_[self._indexHbeta]  = ampl[0] / self.Ralphabeta      * self.extinctionFactor(self.linewave[self._indexHbeta],
                                                                                        self.ebmv,self.Rv)
        ampl_[self._indexHgamma] = ampl[0] / self.Ralphagamma     * self.extinctionFactor(self.linewave[self._indexHgamma],
                                                                                        self.ebmv,self.Rv)
        ampl_[self._indexHdelta] = ampl[0] / self.Ralphadelta     * self.extinctionFactor(self.linewave[self._indexHdelta],
                                                                                        self.ebmv,self.Rv)
        ampl_[self._indexHepsilon] = ampl[0] / self.Ralphaepsilon * self.extinctionFactor(self.linewave[self._indexHepsilon],
                                                                                        self.ebmv,self.Rv)

        return np.asarray(ampl_)*global_ampl
    
    def load_extinction(self):
        """  load the Extinction Factor function
        
        Returns
        -------
        void ; loads self.extinctionFactor
        """
        if "extinctionFactor" in dir(self):
            return 
        try:
            from ToolBox.Astro import Extinction
        except:
            print "You are not at the CCIN2P3. Local Extinction.py used"
            import Extinction
                
        self.extinctionFactor = Extinction.extinctionFactor


    # -------------------- #
    # -  Fit Tricks      - #
    # -------------------- #
    def parse_parameters(self,parameters):
        """ Generique function that works only if the model is made
        of emission line amplitudes and velocity+dispersion
        """
        self.ebmv,self.Rv = parameters[2:-2]
        return parameters[:2],parameters[-2],parameters[-1]
    
    # -------------------- #
    # - Allow Minuit Fit - #
    # -------------------- #
    def _minuit_chi2changed_(self,HA,NII,ebmv,Rv,
                       velocity,dispersion):
        """ Stupid function such that minuit and scipy can be used similarly =
        (Give here all the parameter using the freeparameters order)
        """
        parameter = np.asarray([HA,NII,ebmv,Rv,
                    velocity,dispersion])
        
        return self.get_chi2(parameter)
    
# ------------------------------- #
# --   Halpha NII  Model       -- #
# ------------------------------- #
class OpticalLines_Mains( _OpticalLines_,BaseModel  ):
    """
    """
    FREEPARAMETERS = ["HA","NII","OII1","OII2","SII1","SII2",
                       "velocity","dispersion"]
    
    _indexHalpha  = np.argwhere(_OpticalLines_.LINENAMES == "HA")[0][0]
    _indexNIIR    = np.argwhere(_OpticalLines_.LINENAMES == "NII_2")[0][0]
    _indexNIIB    = np.argwhere(_OpticalLines_.LINENAMES == "NII_1")[0][0]
    
    _indexOII1    = np.argwhere(_OpticalLines_.LINENAMES == "OII1")[0][0]
    _indexOII2    = np.argwhere(_OpticalLines_.LINENAMES == "OII2")[0][0]

    _indexSII1    = np.argwhere(_OpticalLines_.LINENAMES == "SII1")[0][0]
    _indexSII2    = np.argwhere(_OpticalLines_.LINENAMES == "SII2")[0][0]


    # ==================== #
    # =  Model           = #
    # ==================== #
    def read_amplitudes(self,ampl,
                        global_ampl=1.):
        """ reads the Amplitude and return a effective_amplitude array =

        Parameters
        -----------
        
        ampl: [array of float]
            = amplitudes of the lines  =
            Give a list of amplitude for all the Halpha and NII lines
            This Model:
                `ampl` Only Contains 1 NII line amplitude since the 2
                NII line have a fixed ration.
                The Other optical lines are set to 0.
                
        global_ampl: [float]
            = Multiplicative parameter =
            All the amplitude will be multiplied by this.

        Returns
        -------
        array of float ; the amplitudes of the emission lines. See self.LINENAMES
        """
        if len(ampl) != self.nampl:
            raise ValueError("ampl must have exactly %d entry. %d given"%(self.nampl,len(ampl)))

        ampl_ = np.zeros(self._nopticallines)
        ampl_[self._indexHalpha] = ampl[0]
        ampl_[self._indexNIIR]   = ampl[1]
        ampl_[self._indexNIIB]   = ampl[1]*0.34 # bound together
        ampl_[self._indexOII1]   = ampl[2]
        ampl_[self._indexOII2]   = ampl[3]
        ampl_[self._indexSII1]   = ampl[4]
        ampl_[self._indexSII2]   = ampl[5]

        return np.asarray(ampl_)*global_ampl

    # -------------------- #
    # - Allow Minuit Fit - #
    # -------------------- #
    def _minuit_chi2changed_(self,HA,NII,OII1,OII2,SII1,SII2,
                       velocity,dispersion):
        """ Stupid function such that minuit and scipy can be used similarly =
        (Give here all the parameter using the freeparameters order)
        """
        parameter = np.asarray([HA,NII,OII1,OII2,SII1,SII2,
                    velocity,dispersion])
            
        return self.get_chi2(parameter)
        
    
