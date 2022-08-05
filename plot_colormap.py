import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from catchemi import FitParametersNewnsAnderson
from monty.serialization import loadfn
from plot_params import get_plot_params
get_plot_params()

if __name__ == '__main__':

    # Create a contour plot of the energy matrix
    fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)

    data_from_dos_calculation = json.load(open(f"outputs/intermetallics_pdos_moments.json")) 

    # Read in the adsorbate parameters
    ads_parameters = loadfn("CO_parameters.json")
    alpha = ads_parameters['alpha']
    beta = ads_parameters['beta']
    delta0 = ads_parameters['delta0']
    eps_a = ads_parameters['eps_a']
    constant_offset = ads_parameters['constant_offset'] 
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15

    # Spline parameters
    with open(f"outputs/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)

    # Parameters to change delta
    widths = np.linspace(0.2, 12, 15)
    eps_ds = np.linspace(-6, 5.5, 15)
    EPS_VALUES = np.linspace(-30, 30, 1000,) 

    # Store the chemisorption energy
    energy_matrix_hyb = np.zeros((len(widths), len(eps_ds)))
    energy_matrix_ortho = np.zeros((len(widths), len(eps_ds)))
    energy_matrix_tot = np.zeros((len(widths), len(eps_ds)))

    for i, width in enumerate(widths):
        widths_const = np.ones(len(eps_ds)) * width

        kwargs_fit = dict(
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            Delta0_mag = delta0,
            Vsd = np.ones(len(eps_ds)).tolist(), 
            width = widths_const.tolist(), 
            eps_a = eps_a,
            verbose = True,
            store_hyb_energies = True,
        )

        fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)

        # Get the final hybridisation energy
        final_params = alpha + beta + constant_offset
        chemi_energy = fitting_function.fit_parameters(final_params, eps_ds) 
        optimised_hyb = fitting_function.hyb_energy
        optimised_ortho = fitting_function.ortho_energy

        # Store the energy matrix
        energy_matrix_hyb[i, :] = optimised_hyb 
        energy_matrix_ortho[i,:] = optimised_ortho
        energy_matrix_tot[i,:] = chemi_energy

    # Plot the contour
    energy_matrix_hyb = energy_matrix_hyb.T
    energy_matrix_ortho = energy_matrix_ortho.T
    energy_matrix_tot = energy_matrix_tot.T

    cax = ax[0].contourf(widths, eps_ds, energy_matrix_hyb, levels=100)
    cbar = fig.colorbar(cax, ax=ax[0])
    ax[0].contour(widths, eps_ds, energy_matrix_hyb, levels=5, colors='tab:gray', linewidths=0.5)
    # Plot the ortho energies
    cax = ax[1].contourf(widths, eps_ds, energy_matrix_ortho, levels=100)
    cbar = fig.colorbar(cax, ax=ax[1])
    ax[1].contour(widths, eps_ds, energy_matrix_ortho, levels=5, colors='tab:gray', linewidths=0.5)
    # Plot the total energies
    cax = ax[2].contourf(widths, eps_ds, energy_matrix_tot, levels=100)
    cbar = fig.colorbar(cax, ax=ax[2])
    ax[2].contour(widths, eps_ds, energy_matrix_tot, levels=5, colors='tab:gray', linewidths=0.5)

    for _id in data_from_dos_calculation:

        # get the parameters from DFT calculations
        metal = data_from_dos_calculation[_id]['metal_solute']
        width = data_from_dos_calculation[_id]['d_band_width']
        d_band_centre = data_from_dos_calculation[_id]['d_band_centre']
        for a in ax:
            a.plot(width, d_band_centre, 'o', color='k', markersize=2)

    for a in ax:
        a.set_xlabel(r'width (eV)')
        a.set_ylabel(r'$\epsilon_d$ (eV)')
        a.set_xlim(0.2, 12)

    ax[0].set_title(r'Hybridisation')
    ax[1].set_title(r'Orthogonalisation')
    ax[2].set_title(r'Chemisorption')



    fig.savefig('outputs/figure_colormap.png', dpi=300)



