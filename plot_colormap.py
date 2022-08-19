import pickle
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from monty.serialization import loadfn

from catchemi import FitParametersNewnsAnderson
from create_coupling_elements import create_coupling_elements
from plot_params import get_plot_params
get_plot_params()

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

if __name__ == '__main__':
    """Create a series of plots based on varying epsilon_d and 
    w_d freely. The only difference from previous plots is that the
    Vak is proportional to epsilon_d based on the relation in the
    JCP paper and w_d has no influence on the Vak."""

    # ==== Adsorbate parameters ====
    # Each adsorbate has different alpha, beta, delta0
    # and epsilon_a parameters. epsilson_sp min, max
    # are the energy levels corresponding to the 
    # minimum and maximum of the sp-augmentation.
    ads_parameters = loadfn("CO_parameters.json")
    alpha = ads_parameters['alpha']
    beta = ads_parameters['beta']
    delta0 = ads_parameters['delta0']
    eps_a = ads_parameters['eps_a']
    constant_offset = ads_parameters['constant_offset'] 
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    # Running values of the energy
    EPS_VALUES = np.linspace(-30, 30, 1000,) 

    # ==== Transition Metal parameters ====
    # Spline parameters for the transition metals. Currently
    # stored as a pickle file, directly from the JCP paper.
    with open(f"outputs/spline_objects.pkl", 'rb') as f:
        spline_objects = pickle.load(f)
    # Data from Andrew's calculations.
    data_from_dos_calculation = json.load(open(f"outputs/intermetallics_pdos_moments.json")) 
    # Data from the LMTO calculation (for the transition metals).
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    # Each row has a different range of epsilon_d and w_d.
    minmax_parameters = json.load(open('outputs/minmax_parameters.json'))
    # s is a bulk parameter.
    s_data = data_from_LMTO['s']
    # Also need the Anderson band width (bulk parameter).
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    
    # ==== Plot parameters for the 2D plot ====
    GRID_SIZE = 7
    # Create a different subplot for each row and a
    # a different plot for each type of energy. This 
    # will be an imshow 2D plot.
    figsize = (10,3)
    figh, axh = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    figo, axo = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    figc, axc = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    # Width is going to be a free parameter in this
    # model. We do not know if it is or isn't, but we do know
    # that each contour along its variation corresponds to a
    # type of metal (intermetallics, transition metals, or others...)
    width_values = np.linspace(0.3, 6, GRID_SIZE)

    # Iterate over the rows of the transition metals that lead
    # each row has different epsilon_d, w_d and other parameters.
    # This variation allows us to make separate figures for the 
    # different rows, for chemisorption, ortho and hybridisation.
    for j, metal_row in enumerate([FIRST_ROW, SECOND_ROW, THIRD_ROW]):

        # Store the parameters for the different rows.
        parameters_metal = defaultdict(list)

        # get the metal fitting parameters
        s_fit = spline_objects[j]['s']
        Delta_anderson_fit = spline_objects[j]['Delta_anderson']
        wd_fit = spline_objects[j]['width']
        eps_d_fit = spline_objects[j]['eps_d']

        filling_min, filling_max = minmax_parameters[str(j)]['filling']
        filling_range = np.linspace(filling_max, filling_min, GRID_SIZE)
        eps_d_min, eps_d_max = minmax_parameters[str(j)]['eps_d']
        eps_d_range = np.linspace(eps_d_min, eps_d_max, GRID_SIZE)

        # Store the energies for each row.
        energy_matrix_hyb = np.zeros((GRID_SIZE, GRID_SIZE))
        energy_matrix_ortho = np.zeros((GRID_SIZE, GRID_SIZE))
        energy_matrix_tot = np.zeros((GRID_SIZE, GRID_SIZE))

        # Iterate over the filling range to get the required
        # epsilon_d values, each epsilon_d value corresponds to 
        # a transition metal w_d, which is going to be the 
        # contour line. The single atom alloys are most likely
        # not going to be on this contour line.
        for k, filling in enumerate(filling_range):
            # Continuous setting of parameters for each 
            # continous variation of the metal
            width = wd_fit(filling) 
            eps_d = eps_d_fit(filling) 
            Vsdsq = create_coupling_elements(s_metal=s_fit(filling),
                                            s_Cu=s_data['Cu'],
                                            anderson_band_width=Delta_anderson_fit(filling),
                                            anderson_band_width_Cu=anderson_band_width_data['Cu'],
                                            r=s_fit(filling),
                                            r_Cu=s_data['Cu'],
                                            normalise_by_Cu=True,
                                            normalise_bond_length=True
                                            )
            Vsd = np.sqrt(Vsdsq)
            parameters_metal['Vsd'].append(Vsd)
            parameters_metal['eps_d'].append(eps_d)
            parameters_metal['width'].append(width)
            parameters_metal['filling'].append(filling)
            parameters_metal['no_of_bonds'] = np.ones(GRID_SIZE)
        
        # We will iterate over the different epsilon_d values
        # varying around the different width values to generate
        # a 2D plot. 
        # epsilon_d is determined by the filling 
        for indexw, width in enumerate(width_values): 
            # width is a free parameter
            widths_fixed = np.ones(GRID_SIZE) * width
            widths_fixed = widths_fixed.tolist()
            kwargs_fit = dict(
                eps_sp_min = EPS_SP_MIN,
                eps_sp_max = EPS_SP_MAX,
                eps = EPS_VALUES,
                Delta0_mag = delta0,
                Vsd = parameters_metal['Vsd'],
                width = widths_fixed, 
                eps_a = eps_a,
                verbose = True,
                store_hyb_energies = True,
                # precision = 50,
            )
            fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)
                
            # Get the energies
            final_params = alpha + beta + constant_offset
            chemi_energy = fitting_function.fit_parameters(final_params, parameters_metal['eps_d']) 
            optimised_hyb = fitting_function.hyb_energy
            optimised_ortho = fitting_function.ortho_energy

            # Store the energy matrix
            energy_matrix_hyb[indexw, :] = optimised_hyb 
            energy_matrix_ortho[indexw,:] = optimised_ortho
            energy_matrix_tot[indexw,:] = chemi_energy

        # Plot the contour
        energy_matrix_hyb = energy_matrix_hyb.T
        energy_matrix_ortho = energy_matrix_ortho.T
        energy_matrix_tot = energy_matrix_tot.T

        # Plot the contour lines for the different rows.
        cax = axh[j].contourf(width_values, parameters_metal['eps_d'], energy_matrix_hyb, levels=100)
        cbar = figh.colorbar(cax, ax=axh[j])
        axh[j].contour(width_values, parameters_metal['eps_d'], energy_matrix_hyb, levels=5, colors='tab:gray', linewidths=0.5)

        # Plot the ortho energies
        cax = axo[j].contourf(width_values, parameters_metal['eps_d'], energy_matrix_ortho, levels=100)
        cbar = figo.colorbar(cax, ax=axo[j])
        axo[j].contour(width_values, parameters_metal['eps_d'], energy_matrix_ortho, levels=5, colors='tab:gray', linewidths=0.5)

        # Plot the total energies
        cax = axc[j].contourf(width_values, parameters_metal['eps_d'], energy_matrix_tot, levels=100)
        cbar = figc.colorbar(cax, ax=axc[j])
        axc[j].contour(width_values, parameters_metal['eps_d'], energy_matrix_tot, levels=5, colors='tab:gray', linewidths=0.5)

        # Plot the epsilon_d and width values
        axh[j].plot(parameters_metal['width'], parameters_metal['eps_d'], 'o', color='k', markersize=2)
        axo[j].plot(parameters_metal['width'], parameters_metal['eps_d'], 'o', color='k', markersize=2)
        axc[j].plot(parameters_metal['width'], parameters_metal['eps_d'], 'o', color='k', markersize=2)

    # Plot the points from Andrew's calculation
    for _id in data_from_dos_calculation:
        # get the parameters from DFT calculations
        metal = data_from_dos_calculation[_id]['metal_solute']
        width = data_from_dos_calculation[_id]['d_band_width']
        d_band_centre = data_from_dos_calculation[_id]['d_band_centre']
        for ax in [axc, axo, axh]:
            # Choose the index of the plot based on the row number
            if metal in FIRST_ROW:
                index = 0
            elif metal in SECOND_ROW:
                index = 1
            elif metal in THIRD_ROW:
                index = 2
            ax[index].plot(width, d_band_centre, '*', color='k', markersize=2)

    # Label the axes
    for ax in [axc, axo, axh]:
        for a in ax:
            a.set_xlabel('Width (eV)')
            a.set_ylabel('$\epsilon_d$')

            # Set the x-axis limits and the y-axis limits
            # for the different plots
             


    # Save the figure
    figc.savefig('outputs/chemi_energy.png', dpi=300)
    figo.savefig('outputs/ortho_energy.png', dpi=300)
    figh.savefig('outputs/hyb_energy.png', dpi=300)