# modefit

API to fit basic functions thought your data. The core of the module is to be able to transparently use scipy or minuit are following the same rules: set boundaries, give guesses, or even fix any paramters of your model.

Currently, the following models are implemented:
* **binormal step** (`stepfit`). It has 4 parameters: mean_a, mean_b, sigma_a, sigma_b, which are the mean and the dispersion (sigma) of the normal distributions a and b, respectively. Each datapoint can have a probability `proba` (1-`proba`) to belong to the group "a"  ("b"). 

* **Hubblizer** (`get_hubblefit`). Fit the SN standardization around the Hubble diagram (no cosmology fit). This code allows us you to use any parameter to standardize the SN magnitude.
It includes covariance (if given) between the standardization paramaters (i.e. between x1 and c in the `SALT2` framework). In addition, an iterative process enables to retrieve the intrinsic dispersion of the SN magnitude.

* **PolynomeFit** (`get_polyfit`). Fit any degree polynome (legendre or simple) to you dataset (x, y, dy).

## Dependencies

* iminuit (>1.1)
* astrobject (>=0.4)
