# modefit

Fit your data !

The concept of this API is to be able to fit your data using minuit or scipy as will as to run Markov Chain Mote Carlo 
using the same object. 

Once a fitter object is loaded: (see below) simply:
   - use `fit()` to fit the data. See the results in `fitvalues`
   - use `run_mcmc()` to run the markov chain monte carlo. Best values in `mcmc.derived_values`
   
The fitter objects also have a `show()` method in which you can choose to show the best fit or the mcmc samples on top on the data.


### Current Models
Currently, the following models are implemented:
* **binormal step** (`stepfit`). It has 4 parameters: mean_a, mean_b, sigma_a, sigma_b, which are the mean and the dispersion (sigma) of the normal distributions a and b, respectively. Each datapoint can have a probability `proba` (1-`proba`) to belong to the group "a"  ("b"). 

* **Hubblizer** (`get_hubblefit`). Fit the SN standardization around the Hubble diagram (no cosmology fit). This code allows us you to use any parameter to standardize the SN magnitude.
It includes covariance (if given) between the standardization paramaters (i.e. between x1 and c in the `SALT2` framework). In addition, an iterative process enables to retrieve the intrinsic dispersion of the SN magnitude.

* **PolynomeFit** (`get_polyfit`). Fit any degree polynome (legendre or simple) to you dataset (x, y, dy).

## Dependencies

* iminuit (>1.1)
* astrobject (>=0.4)
* emcee (>=2.0)
