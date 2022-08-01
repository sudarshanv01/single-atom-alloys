"""Example of routine to fit the model."""
import json
import yaml
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize, least_squares, leastsq, curve_fit
from scipy import odr
import matplotlib.pyplot as plt
from ase import units
from catchemi import NewnsAndersonLinearRepulsion, FitParametersNewnsAnderson

FIRST_ROW   = [ 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
SECOND_ROW  = [ 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
THIRD_ROW   = [ 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl'] 

def create_coupling_elements(s_metal, s_Cu, 
    anderson_band_width, anderson_band_width_Cu, 
    r=None, r_Cu=None, normalise_bond_length=False,
    normalise_by_Cu=True):
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

if __name__ == '__main__':
    """Determine the fitting parameters for a particular adsorbate."""

    # Choose a sequence of adsorbates
    ADSORBATES = ['CO']
    EPS_A_VALUES = [ [-7, 2.5] ] # eV
    EPS_VALUES = np.linspace(-30, 10, 1000)
    EPS_SP_MIN = -15
    EPS_SP_MAX = 15
    CONSTANT_DELTA0 = 0.1
    print(f"Fitting parameters for adsorbate {ADSORBATES} with eps_a {EPS_A_VALUES}")

    # get the width and d-band centre parameters
    # The moments of the density of states comes from a DFT calculation 
    # and the adsorption energy is from scf calculations of the adsorbate
    # at a fixed distance from the surface.
    data_from_dos_calculation = json.load(open(f"outputs/elementals_pdos_moments.json")) 
    data_from_energy_calculation = json.load(open(f"outputs/elementals_ads_energy.json"))
    data_from_LMTO = json.load(open('inputs/data_from_LMTO.json'))
    # Parse data from the LMTO calculation
    s_data = data_from_LMTO['s']
    anderson_band_width_data = data_from_LMTO['anderson_band_width']
    Vsdsq_data = data_from_LMTO['Vsdsq']
    no_of_bonds = 1. 
    site = 'ontop' # choose between hollow and ontop

    # Plot the fitted and the real adsorption energies
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3), constrained_layout=True)
    ax.set_xlabel('DFT energy (eV)')
    ax.set_ylabel('Chemisorption energy (eV)')
    ax.set_title(f'{ADSORBATES}* with $\epsilon_a=$ {EPS_A_VALUES} eV')

    # simulatenously iterate over ADSORBATES and EPS_A_VALUES
    for i, (adsorbate, eps_a) in enumerate(zip(ADSORBATES, EPS_A_VALUES)):
        print(f"Fitting parameters for adsorbate {adsorbate} with eps_a {eps_a}")
        # Store the parameters in order of metals in this list
        parameters = defaultdict(list)
        # Store the final DFT energies
        dft_energies = []
        metals = []

        for _id in data_from_energy_calculation:

            # get the parameters from DFT calculations
            metal = data_from_dos_calculation[_id]['metal_solute']
            width = data_from_dos_calculation[_id]['d_band_width']
            parameters['d_band_width'].append(width)
            d_band_centre = data_from_dos_calculation[_id]['d_band_centre']
            parameters['d_band_centre'].append(d_band_centre)

            # get the parameters from the energy calculations
            # If it is a sampled calculation, choose the one with
            # the lowest energy.
            adsorption_energy = data_from_energy_calculation[_id]
            if isinstance(adsorption_energy, list):
                dft_energies.append(np.min(adsorption_energy))
            else:
                dft_energies.append(adsorption_energy)
            
            # Get the bond length from the LMTO calculations
            bond_length = data_from_LMTO['s'][metal]*units.Bohr
            bond_length_Cu = data_from_LMTO['s']['Cu']*units.Bohr 
            # Create the coupling element for the solute atom
            Vsdsq = create_coupling_elements(s_metal=s_data[metal],
                s_Cu=s_data['Cu'],
                anderson_band_width=anderson_band_width_data[metal],
                anderson_band_width_Cu=anderson_band_width_data['Cu'],
                r=bond_length,
                r_Cu=bond_length_Cu,
                normalise_bond_length=True,
                normalise_by_Cu=True)

            # Report the square root
            Vsd = np.sqrt(Vsdsq)
            parameters['Vsd'].append(Vsd)

            # Get the metal filling
            filling = data_from_LMTO['filling'][metal]
            parameters['filling'].append(filling)

            # Store the order of the metals
            metals.append(metal)

            # Get the number of bonds based on the 
            # DFT calculation
            parameters['no_of_bonds'].append(no_of_bonds)

        # Prepare the class for fitting routine 
        kwargs_fit = dict(
            eps_sp_min = EPS_SP_MIN,
            eps_sp_max = EPS_SP_MAX,
            eps = EPS_VALUES,
            Delta0_mag = CONSTANT_DELTA0,
            Vsd = parameters['Vsd'],
            width = parameters['d_band_width'],
            eps_a = eps_a,
            verbose = True,
            no_of_bonds = parameters['no_of_bonds'],
        )
        fitting_function =  FitParametersNewnsAnderson(**kwargs_fit)

        # Decide on the number of initial guesses based on the
        # number of eps_a values per adsorbate.
        if isinstance(eps_a, list):
            initial_guess = [ [ 0.01 ]*len(eps_a),
                            [ 0.6 ] * len(eps_a),
                            [ 0.1 ] * len(eps_a) ]
        elif isinstance(eps_a, float) or isinstance(eps_a, int):
            initial_guess = [ [ 0.01 ], [ 0.6 ], [ 0.1 ] ]
        # Flatten the initial guess list
        try:
            initial_guess = [item for sublist in initial_guess for item in sublist]
        except TypeError:
            # If already a list of floats, do nothing
            pass
        
        print('Initial guess: ', initial_guess)

        # Finding the fitting parameters
        data = odr.RealData(parameters['d_band_centre'], dft_energies)
        fitting_model = odr.Model(fitting_function.fit_parameters)
        fitting_odr = odr.ODR(data, fitting_model, initial_guess)
        fitting_odr.set_job(fit_type=2)
        output = fitting_odr.run()

        # Get the final hybridisation energy
        optimised_hyb = fitting_function.fit_parameters(output.beta, parameters['d_band_centre'])

        # plot the parity line
        x = np.linspace(np.min(dft_energies)-0.6, np.max(dft_energies)+0.6, 2)
        ax.plot(x, x, '--', color='tab:grey', linewidth=1)
        # Fix the axes to the same scale 
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(x), np.max(x))

        texts = []
        for j, metal in enumerate(metals):
            # Choose the colour based on the row of the TM
            if metal in FIRST_ROW:
                colour = 'red'
            elif metal in SECOND_ROW:
                colour = 'orange'
            elif metal in THIRD_ROW:
                colour = 'green'
            ax.plot(dft_energies[j], optimised_hyb[j], 'o', color=colour)
            texts.append(ax.text(dft_energies[j], optimised_hyb[j], metal, color=colour, ))

        ax.set_aspect('equal')

        # Write out the fitted parameters as a json file
        json.dump({
            'alpha': abs(output.beta[0]),
            'beta': abs(output.beta[1]),
            'delta0': CONSTANT_DELTA0, 
            'constant_offset': output.beta[2],
            'eps_a': eps_a,
        }, open(f'{adsorbate}_parameters.json', 'w'))

    fig.savefig(f'fitting_example.png', dpi=300)