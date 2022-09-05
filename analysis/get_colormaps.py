"""Plot the variation of the energy components with metal."""

import os

import json

import numpy as np

from monty.serialization import loadfn, dumpfn

import matplotlib.pyplot as plt

from chemisorption_model_simple import SemiEllipseHypothetical

from plot_params import get_plot_params

from collections import defaultdict

get_plot_params()

if __name__ == "__main__":
    """Store the colormap of the energy components for all the
    different metals considered in this study. Vary the d-band
    centre and the width of these materials to simulate the
    effects of alloying / doping on the energy components."""

    # Adsorbate parameters from the `run_model.py` script.
    ADS_PARAMETERS = "inputs/fitting_parameters.json"

    # Get the Vsd parameters
    with open("inputs/vsd_data.json", "r") as handle:
        vsd_data = json.load(handle)

    # All output goes into output_dict
    output_dict = defaultdict(lambda: defaultdict(list))

    # These are all the metals that were used in the study.
    METALS = ["Rh", "Ir", "Pd", "Pt", "Cu"]
    output_dict["METALS"] = METALS

    GRID_SIZE = 20
    eps_d_list = np.linspace(-4, 0, GRID_SIZE)
    w_d_list = np.linspace(0.1, 5, GRID_SIZE)
    eps = np.linspace(-20, 20, 100000)

    # Store the metal parameters
    output_dict["eps_d_list"] = eps_d_list.tolist()
    output_dict["w_d_list"] = w_d_list.tolist()

    for index_m, metal in enumerate(METALS):

        Vsd = vsd_data[metal]
        method = SemiEllipseHypothetical(
            json_params=ADS_PARAMETERS,
            eps_d_list=eps_d_list,
            w_d_list=w_d_list,
            eps=eps,
            Vsd=Vsd,
        )

        # Get the energy components for each metal.
        energy_components = method.generate_meshgrid()

        # Plot the energy components as contourf plots.
        e_hyb, e_ortho, e_chem, occupancy = energy_components

        output_dict["e_hyb"][metal] = e_hyb.tolist()
        output_dict["e_ortho"][metal] = e_ortho.tolist()
        output_dict["e_chem"][metal] = e_chem.tolist()
        output_dict["occupancy"][metal] = occupancy.tolist()

    # Save the output dictionary
    dumpfn(output_dict, "outputs/colormaps.json")
