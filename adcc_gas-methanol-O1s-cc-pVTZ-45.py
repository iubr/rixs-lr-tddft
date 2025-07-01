import xray_help
import pyscf
import numpy as np
import basis_set_exchange
import h5py
import time as tm

start = tm.time()

hartree_in_eV = xray_help.hartree_in_ev()

# Path to the molecular xyz file and basis set
folder = "opt_geometries_for_exp/" 
name = "gas-methanol-PBE0_def2-tzvp"
short_name = "gas-methanol-O1s"
basis_set_label = "cc-pVTZ"

# ADC RIXS settings
level = "adc3"
level_cvs = "cvs-adc2x"
cvs_space = [0]
edge = "O1s"
valence_states = 25
core_states = 25

outname = ( short_name + "_" + edge + "_" 
            + level + "_" + level_cvs + basis_set_label )
outfile = h5py.File(outname + ".h5", "w")

basis_set = {}

# Get the unique atomic symbols
symbols = xray_help.get_symbols(folder + name + ".xyz")
print(symbols)

for symbol in symbols:
    if "def2" in basis_set_label or "cc" in basis_set_label:
        if symbol in ['H']:
            basis = pyscf.gto.load(
                basis_set_exchange.api.get_basis("cc-pVDZ", augment_diffuse=0,
                                                 elements=symbol, fmt="nwchem"),
                                                 symbol)
        else:
            basis = pyscf.gto.load(
                basis_set_exchange.api.get_basis(basis_set_label, augment_diffuse=0,
                                                 elements=symbol, fmt="nwchem"),
                                                 symbol)
    else:
        basis = pyscf.gto.load(
                basis_set_exchange.api.get_basis(basis_set_label, augment_diffuse=0,
                                                 elements=symbol, fmt="nwchem"),
                                                 symbol)

    basis_set[symbol] = basis

# Molecule and SCF reference state
molecule = xray_help.build_molecule(folder + name + ".xyz", basis=basis_set)
hf_ref = xray_help.run_scf_ground_state(molecule, conv_tol=1e-12, verbose=4)

print("\nFinished SCF reference. Starting ADC...\n")

# ADC for the valence-excited states
adc_results = xray_help.absorption_spectrum_adc(hf_ref, adc_level=level,
                                                nsinglets=valence_states,
                                                conv_tol=1e-6)
adc_energies = adc_results.excitation_energy
adc_osc = adc_results.oscillator_strength
outfile.create_dataset("valence_excitation_energies_eV",
                        data = adc_energies * hartree_in_eV)
outfile.create_dataset("valence_oscillator_strengths", data = adc_osc)

print(adc_results.describe())

# CVS-ADC for the core-excited states
cvs_adc_results = xray_help.absorption_spectrum_adc(hf_ref, adc_level=level_cvs,
                                                    nsinglets=core_states,
                                                    cvs_space=cvs_space,
                                                    conv_tol=1e-6)

cvs_energies = cvs_adc_results.excitation_energy
cvs_osc = cvs_adc_results.oscillator_strength
outfile.create_dataset("core_excitation_energies_eV",
                        data = cvs_energies * hartree_in_eV)
outfile.create_dataset("core_oscillator_strengths", data = cvs_osc)

# RIXS settings for first XAS resonance
# Find the first resonance with reasonably large oscillator strength
gamma = 0.00588
omega = -1
for i, osc in enumerate(cvs_osc):
    if osc > 1e-3:
        omega = cvs_energies[i]
        break
if omega < 0:
    print("\n\nERROR ERROR ERROR")
    print(cvs_energies)
    raise ValueError("Could not find an XAS peak with large enough osc. strength!")

print("\nCalculating RIXS for the first XAS resonance at %.2f eV."
       % (omega * hartree_in_eV))
print("Lifetime broadening parameter gamma: %.7f eV." % (gamma * hartree_in_eV))

theta = 45
theta_rad = theta * np.pi / 180.0

# Calculate RIXS
rixs_results = xray_help.calculate_rixs_spectrum(hf_ref, adc_results, cvs_adc_results,
                                                   core_orbitals=cvs_space, omega=omega,
                                                   gamma=gamma,
                                                   valence_interval=(0, 25),
                                                   theta=theta_rad)

# Save results
outfile.create_dataset("lifetime_broadening_param_fwhm_eV",
                        data = gamma * hartree_in_eV)
outfile.create_dataset("lifetime_broadening_param_hwhm_eV",
                        data = 0.5 * gamma * hartree_in_eV)
outfile.create_dataset("incoming_photon_energy_eV", data = omega * hartree_in_eV)
outfile.create_dataset("outgoing_photon_energies_eV", data = np.array(rixs_results["outgoing_photon_energies_ev"]))
outfile.create_dataset("energy_loss_eV", data = np.array(rixs_results["energy_loss_ev"]))

label_angle = "rixs_strengths_theta_%d" % theta
outfile.create_dataset(label_angle, data = rixs_results["rixs_scattering_cross_sections"])

ncore = len(cvs_energies)
nval = len(adc_energies)

for c in range(ncore):
    label = "S0_to_C%d_dipole_mom" % (c + 1)
    outfile.create_dataset(label, rixs_results["gs_to_core_transition_dipole_moments"][c])

for s in range(nval):
    for c in range(ncore):
        label = "S%d_to_C%d_dipole_mom" % (s + 1, c + 1)
        outfile.create_dataset(label,
                       rixs_results["core_to_valence_transition_dipole_moments"][s][c])

outfile.close()

print("\nThe calculation took %.2f min." % ((tm.time() - start)/60.0))
