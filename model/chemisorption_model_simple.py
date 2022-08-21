"""Perform a simple calculation to get the chemisorption energy."""
import os
import logging
import json
from typing import Dict, List, Tuple

from collections import defaultdict

import numpy as np
import numpy.typing as nptyp
import scipy

import matplotlib.pyplot as plt

from plot_params import get_plot_params

get_plot_params()


class SimpleChemisorption:
    """Base class to perform a simple calculation
    to get the chemisorption energy."""

    debug_dir = "debug"
    INTEGRAL_NOISE = 0.3

    def __init__(
        self,
        dft_dos: nptyp.ArrayLike,
        dft_energy_range: nptyp.ArrayLike,
        Vak: float,
        Sak: float,
        Delta0: float,
        eps_a: float,
    ):
        self.dft_dos = dft_dos
        self.eps = dft_energy_range
        self.Vak = Vak
        self.Sak = Sak
        self.Delta0 = Delta0
        self.eps_a = eps_a

        self._validate_inputs()
        # Get the quantities that will be used again and
        # again in the calculation.
        self.get_Delta()
        self.get_Lambda()

    def _validate_inputs(self):
        """Throw up assertion errors if the inputs
        are not in accordance with what is expected."""
        # Ensure that the integral of the density of states is 1.
        # That is, the d-projected density of states is normalised
        # to 1.
        integral_d_dos = np.trapz(self.dft_dos, self.eps)
        assert np.allclose(integral_d_dos, 1.0)
        assert self.Sak <= 0.0, "Sak must be negative or zero."
        assert self.Vak >= 0.0, "Vak must be positive"

        # Make everything a numpy array
        self.dft_dos = np.array(self.dft_dos)
        self.eps = np.array(self.eps)

        # Delta0 must be a float and greater than 0
        assert isinstance(self.Delta0, float), "Delta0 must be a float"
        assert self.Delta0 >= 0.0, "Delta0 must be positive"

        # Create debug directory if it does not exist
        if not os.path.exists(self.debug_dir):
            os.mkdir(self.debug_dir)

    def get_Delta(self) -> nptyp.ArrayLike:
        """Get Vak by multiplying the d-density of
        states by the following relations:
        Delta = pi * Vak**2 * rho_d
        where rho_d is the DFT (normalised) density
        of states.
        """
        self.Delta = np.pi * self.Vak**2 * self.dft_dos
        return self.Delta

    def get_Lambda(self) -> nptyp.ArrayLike:
        """Get the Hilbert transform of the Delta array."""
        self.Lambda = np.imag(scipy.signal.hilbert(self.Delta + self.Delta0))
        return self.Lambda

    def get_chemisorption_energy(self) -> float:
        """The chemisorption energy is the sum of the
        hybridisation energy and the orthogonalisation energy."""
        # self.get_hybridisation_energy()
        # self.get_orthogonalisation_energy()

        self.E_chemi = self.E_hyb + self.E_ortho
        logging.debug(
            f"Chemisorption energy: {self.E_chemi:0.2f} eV, Hybridisation energy: {self.E_hyb:0.2f} eV, Orthogonalisation energy: {self.E_ortho:0.2f} eV; n_a = {self.n_a:0.2f} e"
        )
        return self.E_chemi

    def get_hybridisation_energy(self) -> float:
        """Get the hybridisation energy on the basis
        of the DFT density of states and the Newns-Anderson
        model energy."""

        # Refresh Delta and Lambda
        self.get_Delta()
        self.get_Lambda()

        ind_below_fermi = np.where(self.eps <= 0)[0]

        # Create the integral and just numerically integrate it.
        integrand_numer = self.Delta[ind_below_fermi] + self.Delta0
        integrand_denom = (
            self.eps[ind_below_fermi] - self.eps_a - self.Lambda[ind_below_fermi]
        )

        arctan_integrand = np.arctan2(integrand_numer, integrand_denom)
        arctan_integrand -= np.pi

        # Make sure that the integral is within limits, otherwise
        # something went wrong with the arctan.
        assert np.all(arctan_integrand <= 0), "Arctan integrand must be negative"
        assert np.all(
            arctan_integrand >= -np.pi
        ), "Arctan integrand must be greater than -pi"

        E_hyb = np.trapz(arctan_integrand, self.eps[ind_below_fermi])

        E_hyb *= 2
        E_hyb /= np.pi

        E_hyb -= 2 * self.eps_a

        self.E_hyb = E_hyb

        logging.debug(
            f"Hybridisation energy: {self.E_hyb:0.2f} eV for eps_a: {self.eps_a:0.2f} eV"
        )
        try:
            assert self.E_hyb <= 0.0, "Hybridisation energy must be negative"
        except AssertionError as e:
            # Plot the function of the energy_integrand
            # fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
            # ax.plot(
            #     self.eps[ind_below_fermi],
            #     integrand_numer / integrand_denom,
            #     color="k",
            #     label="integral",
            # )
            # ax.fill_between(
            #     self.eps[ind_below_fermi],
            #     0,
            #     integrand_numer / integrand_denom,
            #     color="k",
            #     alpha=0.2,
            # )
            # # Plot the components of the integral as well
            # ax2 = ax.twinx()
            # ax2.plot(
            #     self.eps[ind_below_fermi],
            #     self.Delta[ind_below_fermi],
            #     color="C0",
            #     label="$\Delta$",
            # )
            # ax2.plot(
            #     self.eps[ind_below_fermi],
            #     self.Delta0 * np.ones(len(self.eps[ind_below_fermi])),
            #     color="C1",
            #     label="$\Delta_0$",
            # )
            # ax2.plot(
            #     self.eps[ind_below_fermi],
            #     self.Lambda[ind_below_fermi],
            #     color="C2",
            #     label="$\Lambda$",
            # )
            # ax2.plot(
            #     self.eps[ind_below_fermi],
            #     self.eps[ind_below_fermi] - self.eps_a,
            #     color="C3",
            #     label="$\epsilon - \epsilon_a$",
            # )
            # ax.set_xlabel("Energy (eV)")
            # ax.set_ylabel("Integrand")
            # ax2.set_ylabel("Parameters")
            # ax2.set_ylim([np.min(self.Lambda), np.max(self.Delta)])
            # fig.savefig(os.path.join(self.debug_dir, "integrand.png"), dpi=300)

            if self.E_hyb < self.INTEGRAL_NOISE:
                logging.warning(
                    f"Hybridisation energy is very slightly greater than 0 ({self.E_hyb:0.2f} eV). This is probably due to numerical integration errors. Setting it to 0."
                )
                self.E_hyb = 0.0
            else:
                # Stop with assertion
                raise e

        return self.E_hyb

    def get_adsorbate_dos(self) -> nptyp.ArrayLike:
        """Get the adsorbate projected density of states."""
        numerator = self.Delta + self.Delta0
        denominator = (self.eps - self.eps_a - self.Lambda) ** 2
        denominator += (self.Delta + self.Delta0) ** 2
        self.rho_a = numerator / denominator / np.pi
        return self.rho_a

    def get_occupancy(self) -> float:
        """Integrate up rho_a upto the Fermi level."""
        # Integrate up rho_a upto the Fermi level.
        index_ = np.where(self.eps <= 0)[0]
        self.n_a = np.trapz(self.rho_a[index_], self.eps[index_])
        return self.n_a

    def get_filling(self) -> float:
        """Integrate up Delta to get the filling of
        the d-density of states of the metal."""
        denominator = np.trapz(self.Delta, self.eps)
        # Get the index of self.eps that are lower than 0
        index_ = np.where(self.eps < 0)[0]
        numerator = np.trapz(self.Delta[index_], self.eps[index_])
        self.filling = numerator / denominator
        assert self.filling > 0.0, "Filling must be positive"
        return self.filling

    def get_orthogonalisation_energy(self) -> float:
        """Get the orthogonalisation energy on the basis of the
        smeared out two-state equation."""
        self.get_Delta()
        self.get_Lambda()
        self.get_adsorbate_dos()
        self.get_occupancy()
        self.get_filling()
        E_ortho = -2 * (self.n_a + self.filling) * self.Vak * self.Sak
        self.E_ortho = E_ortho
        assert self.E_ortho >= 0.0, "Orthogonalisation energy must be positive"
        return self.E_ortho


class AdsorbateChemisorption(SimpleChemisorption):
    """Perform the adsorbate chemisorption analysis
    based on an arbitrary set of parameters. Expected
    input is data related to Vak and the density of states
    from a DFT calculation.

    This class largely wraps around the SimpleChemisorption class
    to ensure that the lists are dealt with correctly."""

    def __init__(
        self,
        dos_data: Dict,
        Vak_data: Dict,
        Sak_data: Dict,
        Delta0: float,
        eps_a_data: List,
    ):

        self.Delta0 = Delta0

        # Treat each adsorbate separately.
        model_outputs = defaultdict(dict)

        for identifier, dos_dict in dos_data.items():
            # Each parameter here is a unique surface.
            self.dft_dos = dos_dict["dft_dos"]
            self.eps = dos_dict["eps"]
            Vak_list = Vak_data[identifier]
            Sak_list = Sak_data[identifier]

            for i, eps_a in enumerate(eps_a_data):
                # Iterate over the adsorbate
                self.eps_a = eps_a

                # Store the float values of Vak and Sak
                self.Vak = Vak_list[i]
                self.Sak = Sak_list[i]

                self._validate_inputs()

                # Compute the chemisorption energy.
                e_hyb = self.get_hybridisation_energy()
                e_ortho = self.get_orthogonalisation_energy()
                e_chemi = self.get_chemisorption_energy()

                # Store the results.
                model_outputs[identifier][eps_a] = {
                    "hyb": e_hyb,
                    "ortho": e_ortho,
                    "chemi": e_chemi,
                }
        self.model_outputs = model_outputs


class FittingParameters:
    """Given a json file with the dos and energy stored, and
    some information about epsilon_a values, perform the fitting
    procedure to determine alpha, beta and a _single_ constant."""

    def __init__(
        self,
        json_filename: str,
        eps_a_data: List,
        Delta0: float,
    ):
        self.json_filename = json_filename
        self.eps_a_data = eps_a_data
        self.Delta0 = Delta0

    def load_data(self):
        """Load the data from the json file."""
        with open(self.json_filename, "r") as f:
            data = json.load(f)
        self.data = data

    def get_comparison_fitting(self, x: Tuple) -> Tuple[List, List]:
        """Check the goodness of the fit by returning
        both the predicted and actual values."""
        gamma = x[-1]
        # Alpha will be the first len(eps_a_data) parameters
        alpha = x[: len(self.eps_a_data)]
        # Beta will be the rest
        beta = x[len(self.eps_a_data) : -1]

        assert (
            len(alpha) == len(beta) == len(self.eps_a_data)
        ), "Parameters must be of equal length"

        # make sure both alpha and beta are always positive
        alpha = np.abs(alpha)
        beta = np.abs(beta)

        # Prepare inputs to the Model class.
        inputs = defaultdict(lambda: defaultdict(dict))
        inputs["Delta0"] = self.Delta0
        inputs["eps_a_data"] = self.eps_a_data

        for _id in self.data:
            # Parse the relevant quantities from the
            # supplied dictionary.
            pdos = self.data[_id]["pdos"]
            energy_grid = self.data[_id]["energy_grid"]
            Vsd = self.data[_id]["Vsd"]
            # Make Vsd an array
            Vsd = np.array(Vsd)

            # Store the inputs.
            inputs["dos_data"][_id]["dft_dos"] = pdos
            inputs["dos_data"][_id]["eps"] = energy_grid
            inputs["Vak_data"][_id] = np.sqrt(beta) * Vsd
            inputs["Sak_data"][_id] = -alpha * Vsd

        # Get the outputs
        model_energies_class = AdsorbateChemisorption(**inputs)
        model_outputs = model_energies_class.model_outputs

        predicted_energy = []
        actual_energy = []
        for _id in model_outputs:
            # What we will compare against
            ads_energy_DFT = self.data[_id]["ads_energy"]
            actual_energy.append(ads_energy_DFT)

            # Construct the model energy
            model_energy = 0.0
            # Add separately for each eps_a
            for eps_a in model_outputs[_id]:
                model_energy += model_outputs[_id][eps_a]["chemi"]

            # Add a constant parameter gamma
            model_energy += gamma
            predicted_energy.append(model_energy)

        logging.info("Predicted and actual energies parsed.")

        return predicted_energy, actual_energy

    def objective_function(
        self,
        x: Tuple,
    ) -> float:
        """Objective function to be minimised."""
        # Infer the parameters from the length of the
        # input tuple.
        # The constant parmeter must always come last
        gamma = x[-1]
        # Alpha will be the first len(eps_a_data) parameters
        alpha = x[: len(self.eps_a_data)]
        # Beta will be the rest
        beta = x[len(self.eps_a_data) : -1]

        assert (
            len(alpha) == len(beta) == len(self.eps_a_data)
        ), "Parameters must be of equal length"

        # make sure both alpha and beta are always positive
        alpha = np.abs(alpha)
        beta = np.abs(beta)

        # Prepare inputs to the Model class.
        inputs = defaultdict(lambda: defaultdict(dict))
        inputs["Delta0"] = self.Delta0
        inputs["eps_a_data"] = self.eps_a_data

        for _id in self.data:
            # Parse the relevant quantities from the
            # supplied dictionary.
            pdos = self.data[_id]["pdos"]
            energy_grid = self.data[_id]["energy_grid"]
            Vsd = self.data[_id]["Vsd"]
            # Make Vsd an array
            Vsd = np.array(Vsd)

            # Store the inputs.
            inputs["dos_data"][_id]["dft_dos"] = pdos
            inputs["dos_data"][_id]["eps"] = energy_grid
            inputs["Vak_data"][_id] = np.sqrt(beta) * Vsd
            inputs["Sak_data"][_id] = -alpha * Vsd

        # Get the outputs
        model_energies_class = AdsorbateChemisorption(**inputs)
        model_outputs = model_energies_class.model_outputs

        # Compute the RMSE value for the difference between
        # the model and DFT data.
        mean_absolute_error = 0.0
        for _id in model_outputs:
            # What we will compare against
            ads_energy_DFT = self.data[_id]["ads_energy"]
            # Construct the model energy
            model_energy = 0.0
            # Add separately for each eps_a
            for eps_a in model_outputs[_id]:
                model_energy += model_outputs[_id][eps_a]["chemi"]

            # Add a constant parameter gamma
            model_energy += gamma

            # Compute the RMSE
            sq_error = np.abs(model_energy - ads_energy_DFT)
            mean_absolute_error += sq_error

        # Return the RMSE
        mean_absolute_error = mean_absolute_error / len(model_outputs)

        logging.info(
            f"Parameters: {x} leads to mean absolute error: {mean_absolute_error}eV"
        )

        return mean_absolute_error
