"""Plot the final colormaps for the manuscript."""

import json
import numpy as np

from monty.serialization import loadfn, dumpfn

import matplotlib.pyplot as plt

plt.rcParams["contour.negative_linestyle"] = "solid"

from plot_params import get_plot_params_andrew

get_plot_params_andrew()

if __name__ == "__main__":
    """Plot the occupancy as a contour plot and the
    the orthogonalisation plot as a heatmap. Also plot the
    chemisorption and hybridisation energy as a heatmap
    which will be supporting information plots."""

    # Get all the data that we need from the JSON file.
    data = loadfn("outputs/colormaps.json")

    METALS = data["METALS"]
    w_d_list = data["w_d_list"]
    eps_d_list = data["eps_d_list"]

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
    # The colormap for the energy components.
    cmap = plt.get_cmap("cividis")

    # Generate subplots for each components
    dim_x, dim_y = 2, len(METALS) // 2
    dim_y += 1 if len(METALS) % 2 == 1 else 0
    levels = 8
    figc, axc = plt.subplots(dim_x, dim_y, figsize=(6.5, 3.5), constrained_layout=True)
    figo, axo = plt.subplots(dim_x, dim_y, figsize=(6.5, 3.5), constrained_layout=True)
    figh, axh = plt.subplots(dim_x, dim_y, figsize=(6.5, 3.5), constrained_layout=True)
    fighp, axhp = plt.subplots(
        dim_x, dim_y, figsize=(6.5, 3.5), constrained_layout=True
    )

    # figc.suptitle("$E_{\mathrm{chemisorption}}$ (eV)")
    # figo.suptitle("$E_{\mathrm{ortho}}$ (eV)")
    # figh.suptitle("$E_{\mathrm{hybridisation}}$ (eV)")

    for index_m, metal in enumerate(METALS):

        e_hyb = data["e_hyb"][metal]
        e_ortho = data["e_ortho"][metal]
        e_chem = data["e_chem"][metal]
        occupancy = data["occupancy"][metal]

        # Take the gradient of e_hyb along w_d to
        e_hyb_wd_prime = np.gradient(e_hyb, w_d_list, axis=0)

        # Get the index of the plot
        index_x, index_y = index_m % 2, index_m // 2
        caxh = axh[index_x, index_y].contourf(
            w_d_list, eps_d_list, e_hyb, levels=levels, cmap=cmap
        )
        caxc = axc[index_x, index_y].contourf(
            w_d_list, eps_d_list, e_chem, levels=levels, cmap=cmap
        )
        caxo = axo[index_x, index_y].contourf(
            w_d_list,
            eps_d_list,
            e_ortho,
            cmap=cmap,
            levels=levels,
        )
        caxoc = axo[index_x, index_y].contour(
            w_d_list,
            eps_d_list,
            e_hyb,
            # cmap="Greys",
            colors="k",
            linewidths=0.75,
        )
        axo[index_x, index_y].clabel(caxoc, inline=True, fontsize=7)
        axhp[index_x, index_y].contourf(
            w_d_list,
            eps_d_list,
            e_hyb_wd_prime,
            cmap="RdBu",
            levels=levels,
        )

        # Make the metal the title of the plot
        axh[index_x, index_y].set_title(metal)
        axc[index_x, index_y].set_title(metal)
        axo[index_x, index_y].set_title(metal)
        axhp[index_x, index_y].set_title(metal)

        # Add the colorbar to the plot
        figh.colorbar(caxh, ax=axh[index_x, index_y])
        figc.colorbar(caxc, ax=axc[index_x, index_y])
        cbaro = figo.colorbar(caxo, ax=axo[index_x, index_y])
        cbaro.ax.set_ylabel("$E_{\mathrm{ortho}}$ (eV)")
        cbarhp = fighp.colorbar(caxo, ax=axhp[index_x, index_y])
        cbarhp.ax.set_ylabel(
            "$\partial E_{\mathrm{hybridisation}} / \partial w_d$ (eV)"
        )

    # Plot the points that Andrew computed on the same plots
    for type_material in ["intermetallics", "elementals"]:
        marker = "o" if type_material == "intermetallics" else "s"
        color = "tab:pink" if type_material == "intermetallics" else "tab:green"
        axh[-1, -1].plot(
            [],
            [],
            marker,
            color=color,
            label=type_material,
            markeredgecolor="k",
            markeredgewidth=1.0,
        )
        axc[-1, -1].plot(
            [],
            [],
            marker,
            color=color,
            label=type_material,
            markeredgecolor="k",
            markeredgewidth=1.0,
        )
        axo[-1, -1].plot(
            [],
            [],
            marker,
            color=color,
            label=type_material,
            markeredgecolor="k",
            markeredgewidth=1.0,
        )

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
                    width,
                    d_band_centre,
                    marker,
                    color=color,
                    markeredgecolor="k",
                    markeredgewidth=1.0,
                    markersize=3,
                )
                ax[index_x, index_y].set_ylabel("$\epsilon_d$ (eV)")
                ax[index_x, index_y].set_xlabel("$w_d$ (eV)")

    # Delete the last unused subplot
    if len(METALS) % 2 == 1:
        axc[-1, -1].axis("off")
        axo[-1, -1].axis("off")
        axh[-1, -1].axis("off")
    # Plot the legend to show the difference between contour
    # and contourf plots
    axo[-1, -1].plot([], [], "k", linewidth=0.75, ls="-", label="$E_{\mathrm{hyb}}$")
    # Set for the points in the last plot
    axo[-1, -1].legend(loc="upper center", frameon=False)
    axc[-1, -1].legend(loc="upper center", frameon=False)
    axh[-1, -1].legend(loc="upper center", frameon=False)

    # Save the plots
    figc.savefig("plots/energy_components_chemisorption.png", dpi=300)
    figo.savefig("plots/energy_components_ortho.png", dpi=300)
    figh.savefig("plots/energy_components_hyb.png", dpi=300)
    fighp.savefig("plots/energy_components_hyb_prime.png", dpi=300)
