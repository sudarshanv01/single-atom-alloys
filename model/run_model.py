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
    no_of_bonds_list = [1, 1]
    DEBUG = True

    logging.info("Loading data from {}".format(JSON_FILENAME))
    logging.info("Using Delta0 = {} eV".format(DELTA0))
    logging.info("Using eps_a = {} eV".format(EPS_A))
    logging.info("Using No. of bonds = {}".format(no_of_bonds_list))

    if DEBUG:
        logging.info("Running in DEBUG mode.")

    if len(sys.argv) > 1:
        if sys.argv[1] == "restart":
            logging.info("Restarting from previous run.")
            with open(OUTPUT_FILE, "r") as f:
                initial_guess_dict = json.load(f)
                initial_guess = (
                    [initial_guess_dict["alpha"]]
                    + [initial_guess_dict["beta"]]
                    + [initial_guess_dict["gamma"]]
                    + initial_guess_dict["epsilon"]
                )
        else:
            raise ValueError("Invalid argument. Only `restart` is allowed.")
    else:
        logging.info("Starting from scratch.")
        initial_guess = [0.1, 0.1, 0.1, 0.1]

    # Perform the fitting routine to get the parameters.
    fitting_parameters = FittingParameters(
        JSON_FILENAME,
        EPS_A,
        DELTA0,
        DEBUG=DEBUG,
        no_of_bonds_list=no_of_bonds_list,
    )
    fitting_parameters.load_data()
    parameters = scipy.optimize.fmin(
        fitting_parameters.objective_function, x0=initial_guess
    )

    # Store the parameters in a json file
    alpha, beta, gamma, *epsilon = parameters

    alpha = np.abs(alpha)
    beta = np.abs(beta)
    gamma = float(gamma)
    epsilon = np.abs(epsilon).tolist()

    fitted_params = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "epsilon": epsilon,
        "eps_a": EPS_A,
        "delta0": DELTA0,
        "no_of_bonds_list": no_of_bonds_list,
    }

    with open(OUTPUT_FILE, "w") as handle:
        json.dump(fitted_params, handle, indent=4)
