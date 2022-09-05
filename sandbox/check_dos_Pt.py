"""Compare the density of states from different computational setups."""

import logging

import numpy as np
import matplotlib.pyplot as plt

import json


class CompareDOS:
    def __init__(self, filename, who_computed, **kwargs):
        """Compare the density of states from different computational setups."""
        self.filename = filename
        self.who_computed = who_computed
        self.kwargs = kwargs

        self.metal_id_VASP = self.kwargs.get(
            "metal_id_VASP", "623c22830f940d8798ddf8f1"
        )
        self.metal_id_QE = self.kwargs.get("metal_id_QE", "Pt")

        self.read_json()

    def read_json(self):
        """Read the density of states from the file."""
        with open(self.filename, "r") as f:
            data = json.load(f)
        self.data = data

    def get_dos_vasp(self):
        """Get the DOS VASP for Pt."""
        pdos = self.data[self.metal_id_VASP]["pdos"]
        energies = self.data[self.metal_id_VASP]["energy_grid"]
        return energies, pdos

    def get_dos_qe(self):
        """Get the DOS from Quantum Espresso for Pt."""
        energies, pdos, _ = self.data["slab"][self.metal_id_QE]
        return energies, pdos

    def get_dos(self):
        """Depending on the computational setup, get the density of states."""
        if self.who_computed == "Andrew":
            self.get_dos_vasp()
        elif self.who_computed == "Sudarshan":
            self.get_dos_qe()
        else:
            raise ValueError("Unknown computational setup.")


def get_first_moment(dist, energy):
    """Get the first moment of the distribution."""
    return np.sum(dist * energy) / np.sum(dist)


if __name__ == "__main__":
    """Compare the density of states and its different
    moments from Andrew's calculation and the one from
    my JCP paper (done with QE)."""

    logging.basicConfig(level=logging.INFO)

    filename_andrew = "../parse/outputs/elementals_pdos_moments.json"
    filename_sudarshan = (
        "inputs/pdos_PBE_SSSP_precision_gauss_smearing_0.1eV_dos_scf.json"
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    # Andrew's calculation
    dos_andrew = CompareDOS(filename_andrew, "Andrew")
    energies_andrew, pdos_andrew = dos_andrew.get_dos_vasp()

    # Sudarshan's calculation
    dos_sudarshan = CompareDOS(filename_sudarshan, "Sudarshan")
    energies_sudarshan, pdos_sudarshan = dos_sudarshan.get_dos_qe()

    # Normalise the pdos
    pdos_andrew /= np.trapz(pdos_andrew, energies_andrew)
    pdos_sudarshan /= np.trapz(pdos_sudarshan, energies_sudarshan)

    centre_andrew = get_first_moment(pdos_andrew, energies_andrew)
    centre_sudarshan = get_first_moment(pdos_sudarshan, energies_sudarshan)
    logging.info(f"Andrew band centre: {centre_andrew} eV")
    logging.info(f"Sudarshan band centre : {centre_sudarshan} eV")

    # Plot the DOS
    ax.plot(
        energies_andrew,
        pdos_andrew,
        label=r"Andrew, $\epsilon_d=$" + f"{centre_andrew:.2f} eV",
    )
    ax.plot(
        energies_sudarshan,
        pdos_sudarshan,
        label="Sudarshan, $\epsilon_d=$" + f"{centre_sudarshan:.2f} eV",
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (arb. units)")
    ax.legend()
    fig.savefig("outputs/dos_Pt.png", dpi=300)
