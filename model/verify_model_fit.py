"""Verify goodness of fit for the model."""
import logging
import json

import numpy as np
import scipy

import matplotlib.pyplot as plt

from chemisorption_model_simple import FittingParameters

if __name__ == "__main__":
    """Plot the model energies against the DFT
    chemisorption energies to see if the model fit
    is good enough."""

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    JSON_FILENAME = "inputs/intermetallics_pdos_moments.json"
    OUTPUT_FILE = "outputs/fitting_parameters.json"
    DEBUG = True

    with open(OUTPUT_FILE, "r") as f:
        final_params_dict = json.load(f)
        final_params = (
            [final_params_dict["alpha"]]
            + [final_params_dict["beta"]]
            + [final_params_dict["gamma"]]
            + final_params_dict["epsilon"]
        )

    DELTA0 = final_params_dict["delta0"]
    EPS_A = final_params_dict["eps_a"]
    no_of_bonds_list = final_params_dict["no_of_bonds_list"]

    # Perform the fitting routine to get the parameters.
    fitting_parameters = FittingParameters(
        JSON_FILENAME,
        EPS_A,
        DELTA0,
        DEBUG=DEBUG,
        return_extended_output=True,
        no_of_bonds_list=no_of_bonds_list,
    )
    fitting_parameters.load_data()
    mea, output_data = fitting_parameters.objective_function(final_params)

    predicted_energy = output_data["predicted_energy"]
    actual_energy = output_data["actual_energy"]

    # Determine the R^2 value of the fit.
    slope, intercept, r, p, se = scipy.stats.linregress(actual_energy, predicted_energy)
    r_squared = r**2
    logging.info(f"R^2 value: {r_squared}")

    # Get the id numbers of energies
    id_numbers = output_data["id_order"]
    # Species string
    species = output_data["species_string"]
    logging.info("{}".format(species))

    # Data dict for the intermetallic species.
    data = fitting_parameters.data

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax.plot(predicted_energy, actual_energy, "o")
    ax.set_xlabel("Model Predicted (eV)")
    ax.set_ylabel("DFT (eV)")
    ax.set_title("Model fit")
    ax.set_aspect("equal")

    # Annotate the plot with species information
    for i, spec in enumerate(species):
        ax.annotate(spec, xy=(predicted_energy[i], actual_energy[i]), fontsize=6)

    # Plot parity plot
    x_par = np.linspace(min(predicted_energy), max(predicted_energy), 100)
    y_par = x_par
    ax.plot(x_par, y_par, "k-")
    fig.savefig("outputs/model_fit.png", dpi=300)
