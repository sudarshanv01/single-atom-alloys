import sys
import logging
import json

import numpy as np
import scipy

from chemisorption_model_simple import FittingParameters

if __name__ == "__main__":
    """Test out the density of states method of the class."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logging.info("Start fitting routine.")

    JSON_FILENAME = "inputs/intermetallics_pdos_moments.json"
    OUTPUT_FILE = "outputs/fitting_parameters.json"
    DELTA0 = 0.1  # eV
    EPS_A = [-7, 2.5]  # For CO*
    DEBUG = True
    logging.info("Loading data from {}".format(JSON_FILENAME))
    logging.info("Using Delta0 = {} eV".format(DELTA0))
    logging.info("Using eps_a = {} eV".format(EPS_A))
    if DEBUG:
        logging.info("Running in DEBUG mode.")

    if len(sys.argv) > 1:
        if sys.argv[1] == "restart":
            logging.info("Restarting from previous run.")
            with open(OUTPUT_FILE, "r") as f:
                initial_guess_dict = json.load(f)
                initial_guess = (
                    initial_guess_dict["alpha"]
                    + initial_guess_dict["beta"]
                    + [initial_guess_dict["gamma"]]
                )
        else:
            raise ValueError("Invalid argument. Only `restart` is allowed.")
    else:
        logging.info("Starting from scratch.")
        initial_guess = [0.1, 0.1] * len(EPS_A) + [0.1]

    # Perform the fitting routine to get the parameters.
    fitting_parameters = FittingParameters(JSON_FILENAME, EPS_A, DELTA0, DEBUG=DEBUG)
    fitting_parameters.load_data()
    parameters = scipy.optimize.fmin(
        fitting_parameters.objective_function, x0=initial_guess
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

    with open(OUTPUT_FILE, "w") as handle:
        json.dump(fitted_params, handle, indent=4)
