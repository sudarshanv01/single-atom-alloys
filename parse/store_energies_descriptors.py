"""Store the energies and band-centres."""
import logging
import os
import json

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

from monty.serialization import loadfn

from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.io.vasp import Vasprun

from ase.dft import get_distribution_moment
from ase import units


def semi_ellipse(energies, eps_d, width, amp):
    """Functional form of a semi-ellipse to allow
    for fitting the density of states."""
    energy_ref = (energies - eps_d) / width
    delta = np.zeros(len(energies))
    for i, eps_ in enumerate(energy_ref):
        if np.abs(eps_) < 1:
            delta[i] = amp * (1 - eps_**2) ** 0.5
    return delta


def create_coupling_elements(
    s_metal: float,
    s_Cu: float,
    anderson_band_width: float,
    anderson_band_width_Cu: float,
    r: float = None,
    r_Cu: float = None,
    normalise_bond_length=False,
    normalise_by_Cu=True,
):
    """Create the coupling elements based on the Vsd
    and r values. The Vsd values are identical to those
    used in Ruban et al. The assume that the bond lengths
    between the metal and adsorbate are the same. Everything
    is referenced to Cu, as in the paper by Ruban et al."""
    Vsdsq = s_metal**5 * anderson_band_width
    Vsdsq_Cu = s_Cu**5 * anderson_band_width_Cu
    if normalise_by_Cu:
        Vsdsq /= Vsdsq_Cu
    if normalise_bond_length:
        assert r is not None
        if normalise_by_Cu:
            assert r_Cu is not None
            Vsdsq *= r_Cu**8 / r**8
        else:
            Vsdsq /= r**8
    return Vsdsq


if __name__ == "__main__":
    """Store the energies and the d-band centres in a json file."""

    logging.basicConfig(level=logging.INFO, filename="parser.log", filemode="w")

    # Path to the vasprun folder
    vaspruns_path = "vaspruns"
    logging.info(f"Reading vaspruns from {vaspruns_path}")

    # Input parameters
    data_from_LMTO = json.load(open("inputs/data_from_LMTO.json"))
    logging.info(f"Reading LMTO input.")

    # Parse data from the LMTO calculation
    s_data = data_from_LMTO["s"]
    anderson_band_width_data = data_from_LMTO["anderson_band_width"]

    for type_calc in ["intermetallics", "elementals"]:

        # Data is stored in different files
        dict_data = loadfn(f"inputs/{type_calc}.json")

        output_data = defaultdict(dict)

        for mp_id in dict_data:

            for ads_id in range(len(dict_data[mp_id]["slabs"])):

                logging.info("Processing {} {}".format(mp_id, ads_id))

                # Make a separate plot for each intermetallic
                fig, ax = plt.subplots(1, 1, figsize=(4, 6), constrained_layout=True)
                ax.set_ylabel("Energy (eV)")
                ax.set_xlabel("Projected DOS")

                # Get the slab id
                _id = dict_data[mp_id]["slabs"][ads_id]["id"]
                _id_co = dict_data[mp_id]["adsorption"][ads_id]["id"]

                # Get the adsorption site
                ads_site = dict_data[mp_id]["adsorption"][ads_id]["ads_site"]

                # Get the adsorption energy
                ads_energy = dict_data[mp_id]["adsorption"][ads_id]["ads_energy"]
                logging.info(f"Adsorption energy for {mp_id} is {ads_energy}")

                # Get the vasprun file
                vr = Vasprun(
                    os.path.join(
                        vaspruns_path, type_calc, "slab", _id + "_vasprun.xml.gz"
                    )
                )
                vr_co = Vasprun(
                    os.path.join(
                        vaspruns_path, type_calc, "ads_slab", _id_co + "_vasprun.xml.gz"
                    )
                )

                # Store the dos
                dos = vr.complete_dos
                structure = dos.structure
                dos_co = vr_co.complete_dos
                structure_co = dos_co.structure

                # Get the distance between the C atom in CO and the adsorption site
                carbon_index = structure_co.indices_from_symbol("C")[0]
                bond_length_co = structure_co.get_distance(carbon_index, ads_site)
                logging.info(f"Bond length: {bond_length_co} Angstrom")

                # Write out the structures
                structure.to(filename=f"outputs/{type_calc}/{mp_id}_structure.cif")

                # Get the bond length from the LMTO calculations
                ads_metal = structure[ads_site].specie.symbol
                bond_length = data_from_LMTO["s"][ads_metal] * units.Bohr
                bond_length_Cu = data_from_LMTO["s"]["Cu"] * units.Bohr

                # Calculate and store the coupling elements
                Vsdsq = create_coupling_elements(
                    s_metal=s_data[ads_metal],
                    s_Cu=s_data["Cu"],
                    anderson_band_width=anderson_band_width_data[ads_metal],
                    anderson_band_width_Cu=anderson_band_width_data["Cu"],
                    r=bond_length,
                    r_Cu=bond_length_Cu,
                    normalise_bond_length=True,
                    normalise_by_Cu=True,
                )

                Vsd = Vsdsq**0.5
                logging.info(f"Coupling element for {ads_metal} is {Vsd}")

                # Get the metal atom to which the CO is bound
                metal_solute = structure[ads_site].specie.symbol

                # Generate a list of all species in the structure
                species = [site.specie.symbol for site in structure]
                species = np.unique(species).tolist()

                # Extract the projected density of states
                pdos = dos.get_site_spd_dos(structure[ads_site])[OrbitalType.d]
                pdos_extract = pdos.get_densities()
                # Extract the energies from the projected density of states
                # and reference to the Fermi level
                energy = pdos.energies - pdos.efermi
                # Extract also the pdos for the CO
                pdos_co = dos_co.get_site_spd_dos(structure_co[carbon_index])[
                    OrbitalType.s
                ]
                pdos_extract_co = pdos_co.get_densities()
                pdos_extract_co = np.array(pdos_extract_co)
                pdos_co = dos_co.get_site_spd_dos(structure_co[carbon_index])[
                    OrbitalType.p
                ]
                pdos_extract_co += pdos_co.get_densities()
                energy_co = pdos_co.energies - pdos_co.efermi

                # If the lowest energy is higher than -20, pad
                # the pdos array with zeros and the energy arrays
                # till -20
                if np.min(energy) > -20:
                    # Pad the energy array with linear spacing
                    # until -20
                    energy_spacing = energy[1] - energy[0]
                    energy_left = np.arange(-20, np.min(energy), energy_spacing)
                    # Pad the pdos array with zeros
                    pdos_left = np.zeros(len(energy_left))
                    # Concatenate the arrays
                    energy = np.concatenate((energy_left, energy))
                    pdos_extract = np.concatenate((pdos_left, pdos_extract))

                # Smear out the projected density of states
                sigma = 0.1  # can change if needed
                diff = [energy[i + 1] - energy[i] for i in range(len(energy) - 1)]
                avgdiff = sum(diff) / len(diff)
                pdos_extract = gaussian_filter1d(pdos_extract, sigma / avgdiff)
                # Do the same smearing for CO
                diff_co = [
                    energy_co[i + 1] - energy_co[i] for i in range(len(energy_co) - 1)
                ]
                avgdiff_co = sum(diff_co) / len(diff_co)
                pdos_extract_co = gaussian_filter1d(pdos_extract_co, sigma / avgdiff_co)

                # Normalise the extracted pdos
                normalising_integral = np.trapz(pdos_extract, energy)
                pdos_extract /= normalising_integral
                normalising_integral_co = np.trapz(pdos_extract_co, energy_co)
                pdos_extract_co /= normalising_integral_co

                # Get the band centres and widths
                center_ase, width_ase = get_distribution_moment(
                    energy, pdos_extract, (1, 2)
                )
                popt, pcov = curve_fit(
                    semi_ellipse, energy, pdos_extract, p0=[center_ase, width_ase, 1]
                )
                centre, width, amp = popt

                # Plot the band centre and width as
                ax.plot(pdos_extract, energy, "-", color="k")
                ax.plot(pdos_extract, energy, ".", color="k", alpha=0.1)
                ax.plot(pdos_extract_co, energy_co, "-", color="tab:blue")
                ax.fill_between(pdos_extract, 0, energy, color="k", alpha=0.2)
                ax.fill_between(
                    pdos_extract_co, 0, energy_co, color="tab:blue", alpha=0.2
                )
                ax.axhline(center_ase, color="tab:red", linestyle="--")
                ax.axhline(center_ase + width_ase, color="tab:blue", linestyle="--")
                ax.axhline(center_ase - width_ase, color="tab:blue", linestyle="--")

                # Generate output_data
                output_data[_id]["d_band_centre"] = centre
                output_data[_id]["d_band_width"] = width
                output_data[_id]["metal_solute"] = metal_solute
                output_data[_id]["pdos"] = pdos_extract.tolist()
                output_data[_id]["energy_grid"] = energy.tolist()
                output_data[_id]["ads_energy"] = ads_energy
                output_data[_id]["Vsd"] = float(Vsd)
                output_data[_id]["species"] = species
                output_data[_id]["coord_num"] = dict_data[mp_id]["adsorption"][0][
                    "coord_num"
                ]
                logging.info(f"Stored {_id}")

                ax.plot(pdos_extract, energy, "-", color="k")
                ax.set_xlabel("Projected DOS")
                ax.set_ylabel("Energy (eV)")
                ax.set_ylim([-20, center_ase + 7])
                # Save the figure
                fig.savefig(os.path.join("plots", type_calc, _id + ".png"))
                plt.close(fig)

        with open(f"outputs/{type_calc}_pdos_moments.json", "w") as handle:
            json.dump(output_data, handle, indent=4)

        logging.info("Total number of structures parsed: {}".format(len(output_data)))
