"""Verify goodness of fit for the model."""
import json

import numpy as np

import matplotlib.pyplot as plt

from chemisorption_model_simple import FittingParameters

if __name__ == "__main__":
    """Plot the model energies against the DFT
    chemisorption energies to see if the model fit
    is good enough."""

    JSON_FILENAME = "inputs/intermetallics_pdos_moments.json"
    OUTPUT_FILE = "outputs/fitting_parameters.json"
    DELTA0 = 0.1  # eV
    EPS_A = [-7, 2.5]  # For CO*

    with open(OUTPUT_FILE, "r") as f:
        final_params_dict = json.load(f)
        final_params = (
            final_params_dict["alpha"]
            + final_params_dict["beta"]
            + [final_params_dict["gamma"]]
        )

    # Perform the fitting routine to get the parameters.
    fitting_parameters = FittingParameters(JSON_FILENAME, EPS_A, DELTA0)
    fitting_parameters.load_data()
    predicted, actual = fitting_parameters.get_comparison_fitting(final_params)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax.plot(predicted, actual, "o")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Model fit")
    ax.set_aspect("equal")
    # Plot parity plot
    x_par = np.linspace(min(predicted), max(predicted), 100)
    y_par = x_par
    ax.plot(x_par, y_par, "k-")
    fig.savefig("outputs/model_fit.png", dpi=300)
