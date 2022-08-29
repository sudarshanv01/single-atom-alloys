"""Plot the variation of the energy components with metal."""

import os

import json

import numpy as np

import matplotlib.pyplot as plt

from chemisorption_model_simple import SemiEllipseHypothetical

from plot_params import get_plot_params

get_plot_params()

if __name__ == "__main__":
    """Plot a colormap of the energy components for all the
    different metals considered in this study. Vary the d-band
    centre and the width of these materials to simulate the
    effects of alloying / doping on the energy components."""

    # Adsorbate parameters from the `run_model.py` script.
    ADS_PARAMETERS = "inputs/fitting_parameters.json"

    # Get the Vsd parameters
    with open("inputs/vsd_data.json", "r") as handle:
        vsd_data = json.load(handle)

    # These are all the metals that were used in the study.
    METALS = ["Rh", "Ir", "Pd", "Pt", "Cu"]

    data_from_dos_calc_intermetallics = json.load(
        open(f"inputs/intermetallics_pdos_moments.json")
    )
    data_from_dos_calc_elementals = json.load(
        open(f"inputs/elementals_pdos_moments.json")
    )

    data_from_dos_calculation = {
        "intermetallics": data_from_dos_calc_intermetallics,
        "elementals": data_from_dos_calc_elementals,
    }

    # Generate subplots for each components
    dim_x, dim_y = 2, len(METALS) // 2
    dim_y += 1 if len(METALS) % 2 == 1 else 0
    figc, axc = plt.subplots(dim_x, dim_y, figsize=(6, 3.5), constrained_layout=True)
    figo, axo = plt.subplots(dim_x, dim_y, figsize=(6, 3.5), constrained_layout=True)
    figh, axh = plt.subplots(dim_x, dim_y, figsize=(6, 3.5), constrained_layout=True)

    figc.suptitle("$E_{\mathrm{chemisorption}}$ (eV)")
    figo.suptitle("$E_{\mathrm{orthogonalisation}}$ (eV)")
    figh.suptitle("$E_{\mathrm{hybridisation}}$ (eV)")

    GRID_SIZE = 25
    eps_d_list = np.linspace(-5, 4, GRID_SIZE)
    w_d_list = np.linspace(0.1, 5, GRID_SIZE)
    eps = np.linspace(-20, 20, 1000)

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
        e_hyb, e_ortho, e_chem = energy_components

        # Get the index of the plot
        index_x, index_y = index_m % 2, index_m // 2
        caxh = axh[index_x, index_y].contourf(eps_d_list, w_d_list, e_hyb, cmap="RdBu")
        caxc = axc[index_x, index_y].contourf(eps_d_list, w_d_list, e_chem, cmap="RdBu")
        caxo = axo[index_x, index_y].contourf(
            eps_d_list, w_d_list, e_ortho, cmap="RdBu"
        )

        # Make the metal the title of the plot
        axh[index_x, index_y].set_title(metal)
        axc[index_x, index_y].set_title(metal)
        axo[index_x, index_y].set_title(metal)

        # Add the colorbar to the plot
        figh.colorbar(caxh, ax=axh[index_x, index_y])
        figh.colorbar(caxc, ax=axc[index_x, index_y])
        figh.colorbar(caxo, ax=axo[index_x, index_y])

    # Plot the points that Andrew computed on the same plots
    for type_material in ["intermetallics", "elementals"]:
        marker = "o" if type_material == "intermetallics" else "*"
        color = "k" if type_material == "intermetallics" else "tab:green"

        for _id in data_from_dos_calculation[type_material]:
            # get the parameters from DFT calculations
            metal = data_from_dos_calculation[type_material][_id]["metal_solute"]
            # Get the index of the plot
            index_m = METALS.index(metal)
            index_x, index_y = index_m % 2, index_m // 2

            width = data_from_dos_calculation[type_material][_id]["d_band_width"]
            d_band_centre = data_from_dos_calculation[type_material][_id][
                "d_band_centre"
            ]

            for ax in [axc, axo, axh]:
                ax[index_x, index_y].plot(
                    d_band_centre, width, marker, color=color, markersize=3
                )

    # Delete the last unused subplot
    if len(METALS) % 2 == 1:
        figc.delaxes(axc[-1, -1])
        figo.delaxes(axo[-1, -1])
        figh.delaxes(axh[-1, -1])

    # Save the plots
    figc.savefig("plots/energy_components_chemisorption.png", dpi=300)
    figo.savefig("plots/energy_components_ortho.png", dpi=300)
    figh.savefig("plots/energy_components_hyb.png", dpi=300)
