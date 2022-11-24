`single-atom-alloys`
-------------------

[![DOI](https://zenodo.org/badge/519363038.svg)](https://zenodo.org/badge/latestdoi/519363038)


This repository serves as a complement to the paper "Free-atom-like d-states Beyond the Dilute Limit of Single-Atom Alloys" [link](https://chemrxiv.org/engage/chemrxiv/article-details/63312eadba8a6d2f525d30b7).


# Requirements

The requirements for the code are listed in `requirements.txt`. In order to download the required packages, run the following command:

    pip install -r requirements.txt


# Navigation

The code is organized in the following directories:

- `parse`: Contains scripts to parse the VASP calculations reported in the manuscript. Further details on computational details can be found in the Methods section of the manuscript.
- `model`: Implementation of the Newns-Anderson model with the effective orthogonalisation term. The model itself is identical to that reported in the paper: [link](https://aip.scitation.org/doi/full/10.1063/5.0096625) and implemented as a python package [here](https://github.com/sudarshanv01/CatChemi). The code in this directory is a modified version of the code in the `CatChemi` package.
- `analysis`: Contains scripts to analyse and plot the results of the model.


Please see the README files in each directory for further details.