`model`
------

This directory contains the `python` script `chemisorption_model_simple.py`, which includes classes to run the model described in the manuscript. It is a simplified version of the python package [here](https://github.com/sudarshanv01/CatChemi).


A brief description of the classes in the script is given below:

* `SimpleChemisorption`: Class to compute the Hybridisation, orthogonalization, and chemisorption energy. It is a useful base class to calculate the chemisorption energy for a given set of parameters.
* `SemiEllipseHypothetical`: Class to construct a semi-ellipse representation give the d-band centers and width. The use-case is to provide a series of d-band centers and widths to compute the chemisorption energy for a fixed coupling element value.   
* `AdsorbateChemisorption`: Class to wrap around `SimpleChemisorption` for many metals.
* `FittingParameters`: Fit alpha and beta to determine the constants that parameterize the model.

Class docstrings provide more details on the methods and attributes of the classes as well as required inputs.