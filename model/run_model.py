import logging
import json

import numpy as np

from chemisorption_model_simple import FittingParameters

if __name__ == "__main__":
    """Test out the density of states method of the class."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logging.info("Start fitting routine.")

    JSON_FILENAME = "outputs/intermetallics_pdos_moments.json"
    DELTA0 = 0.1  # eV
    EPS_A = [-7, 2.5]  # For CO*
    logging.info("Loading data from {}".format(JSON_FILENAME))
    logging.info("Using Delta0 = {} eV".format(DELTA0))
    logging.info("Using eps_a = {} eV".format(EPS_A))

    # Perform the fitting routine to get the parameters.
    fitting_parameters = FittingParameters(JSON_FILENAME, EPS_A, DELTA0)
    fitting_parameters.load_data()
    parameters = scipy.optimize.fmin(
        fitting_parameters.objective_function, x0=[0.1, 0.1, 0.1, 0.1, 0.1]
    )

    # Store the parameters in a json file
    alpha = parameters[: len(EPS_A)]
    beta = parameters[len(EPS_A) : -1]
    gamma = parameters[-1]

    alpha = np.abs(alpha).tolist()
    beta = np.abs(beta).tolist()
    gamma = float(gamma)

    fitted_params = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    with open("outputs/fitting_parameters.json", "w") as handle:
        json.dump(fitted_params, handle)
