# modefit

[![PyPI](https://img.shields.io/pypi/v/modefit.svg?style=flat-square)](https://pypi.python.org/pypi/modefit)

Fit your data !

The concept of this API is to be able to fit your data using minuit or scipy as will as to run Markov Chain Mote Carlo 
using the same object. 

Once a fitter object is loaded: (see below) simply:
   - use `fit()` to fit the data. See the results in `fitvalues`
   - use `run_mcmc()` to run the markov chain monte carlo. Best values in `mcmc.derived_values`
   
The fitter objects also have a `show()` method in which you can choose to show the best fit or the mcmc samples on top on the data.

# Installation

`pip install modefit` (favored)

or 

```bash
git pull https://github.com/MickaelRigault/modefit.git
cd modefit
python setup.py install
```


### Current Models
Currently, the following models are implemented:
* **binormal step** (`stepfit`). It has 4 parameters: mean_a, mean_b, sigma_a, sigma_b, which are the mean and the dispersion (sigma) of the normal distributions a and b, respectively. Each datapoint can have a probability `proba` (1-`proba`) to belong to the group "a"  ("b"). 

* **PolynomeFit** (`get_polyfit`). Fit any degree polynome (legendre or simple) to your dataset (x, y, dy).

* **NormPolynomeFit** (`get_normpolyfit`). Fit any number of gaussian on top of a polynome of any degree (legendre or simple) to your dataset (x, y, dy). 

## Dependencies

* iminuit (>1.1)
* propobject (>=0.1)
* emcee (>=2.0) _not mandatory if only fitting_
