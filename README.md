# modefit

API to fit basic functions thought your data. The core of the module is to be able to transparently use scipy or minuit are following the same rules: set boundaries, give guesses, or even fix any paramters of your model.

Currently, the following models are implemented:
* **binormal step** (step_fitter). It has 4 parameters: mean_a, mean_b, sigma_a, sigma_b, which are the mean and the dispersion (sigma) of normal distributions for the samples a and b, respectively. Each datapoint can have a probability `proba` to belong to the group "a" and so a probability `1-proba` to belong to the group "b". 


## Dependencies

* iminuit (>1.1)
