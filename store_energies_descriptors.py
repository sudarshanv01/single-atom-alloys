"""Store the energies and band-centres."""
import os
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
from scipy.optimize import curve_fit
from monty.serialization import loadfn
from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.io.vasp import Vasprun
from ase.dft import get_distribution_moment

def semi_ellipse(energies, eps_d, width, amp):
    energy_ref = ( energies - eps_d ) / width
    delta = np.zeros(len(energies))
    for i, eps_ in enumerate(energy_ref):
        if np.abs(eps_) < 1:
            delta[i] = amp * (1 - eps_**2)**0.5
    return delta

if __name__ == '__main__':
    """Store the energies and the d-band centres in a json file."""
    # Path to the vasprun folder
    vaspruns_path = "vaspruns"

    intermetallics = loadfn("intermetallics.json")
    elementals = loadfn("elementals.json")

    # Store moments by _id
    moments = defaultdict(dict)

    for mp_id in intermetallics:

        # Get the slab id
        _id =intermetallics[mp_id]["slabs"][0]["id"]

        # Make a separate plot for each intermetallic
        fig, ax = plt.subplots(1, 1, figsize=(4,6), constrained_layout=True)
        ax.set_ylabel('Energy (eV)')
        ax.set_xlabel('Projected DOS')

        # Get the adsorption site
        ads_site = intermetallics[mp_id]["adsorption"][0]["ads_site"]

        # Get the vasprun file
        vr = Vasprun(os.path.join(vaspruns_path, "intermetallics", "slab", _id + "_vasprun.xml.gz"))

        # Store the dos
        dos = vr.complete_dos
        structure = dos.structure 

        # Get the metal atom to which the CO is bound
        metal_solute = structure[ads_site].specie.symbol

        # Extract the projected density of states
        pdos = dos.get_site_spd_dos(structure[ads_site])[OrbitalType.d]

        # Make x and y variables to plot
        pdos_extract = pdos.get_densities()
        energy = pdos.energies - pdos.efermi

        # Get the band centres and widths
        center_ase, width_ase = get_distribution_moment(energy, pdos_extract,(1,2))
        popt, pcov = curve_fit(semi_ellipse, energy, pdos_extract, p0=[center_ase, width_ase, 1])
        centre, width, amp = popt

        # Plot the band centre and width as 
        ax.plot(pdos_extract, energy, '-', color='k')
        ax.fill_between(pdos_extract, 0, energy, color='k', alpha=0.2)
        ax.axhline(center_ase, color='tab:red', linestyle='--')
        ax.axhline(center_ase + width_ase, color='tab:blue', linestyle='--')
        ax.axhline(center_ase - width_ase, color='tab:blue', linestyle='--')

        moments[_id]["d_band_centre"] = centre
        moments[_id]["d_band_width"] = width
        moments[_id]["metal_solute"] = metal_solute

        # Plot semi-ellipse
        ax.plot(semi_ellipse(energy, centre, width, amp), energy, '-', color='tab:green')

        ax.set_ylim([center_ase-7, center_ase+7])
        # Save the figure
        fig.savefig(os.path.join("plots", "intermetallics", _id + ".png"))
    
    with open(f"outputs/pdos_moments.json", 'w') as handle:
        json.dump(moments, handle, indent=4)
