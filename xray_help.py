pyscf_success = False
vlx_success = False
try:
    import pyscf
    from pyscf import tdscf
    pyscf_success = True
except ImportError:
    print("* XRayHelp Warning * PyScf is not available.")
try:
    import adcc
except ImportError:
    print("* XRayHelp Warning * Adcc is not available.")

try:
    import veloxchem as vlx
    from veloxchem.veloxchemlib import hartree_in_ev
    vlx_success = True
except:
    print("* XRayHelp Warning * VeloxChem is not available.")
    def hartree_in_ev():
        return 27.211386245988 

import numpy as np
import time
import copy
import h5py
try:
    from matplotlib import pyplot as plt
    from matplotlib import colormaps as cmaps
except ImportError:
    print("* XRayHelp Warning * Matplotlib is not available.")
try:
    import pandas as pd
except ImportError:
    print("* XRayHelp Warning * Pandas is not available.")
import scipy
from scipy.constants import physical_constants

def get_nocc_mo_coeff(scf_reference_state):
    if vlx_success and pyscf_success:
        if isinstance(scf_reference_state, vlx.ScfRestrictedDriver):
            nocc = int(np.sum(scf_reference_state.scf_tensors['occ_alpha']))
            mo_coeff = scf_reference_state.scf_tensors['C_alpha']
            return nocc, mo_coeff
        else:
            nocc = scf_reference_state.mol.nelec[0]
            mo_coeff = scf_reference_state.mo_coeff
            return nocc, mo_coeff
    elif vlx_success:
        nocc = int(np.sum(scf_reference_state.scf_tensors['occ_alpha']))
        mo_coeff = scf_reference_state.scf_tensors['C_alpha']
        return nocc, mo_coeff
    elif pyscf_success:
        nocc = scf_reference_state.mol.nelec[0]
        mo_coeff = scf_reference_state.mo_coeff
        return nocc, mo_coeff
    else:
        raise ValueError("Neither VeloxChem nor PyScf could be loaded.") 

def get_dipole_moment_integrals(scf_reference_state):
    """ Returns the dipole moment integrals in AO basis.
    """
    if vlx_success and pyscf_success:
        if isinstance(scf_reference_state, vlx.ScfRestrictedDriver):
            molecule = scf_reference_state.task.molecule
            basis = scf_reference_state.task.ao_basis
            dipmom = vlx.compute_electric_dipole_integrals(molecule, basis)
            dipole_moment_integrals_ao = np.array(dipmom)
            return dipole_moment_integrals_ao
        else:
            dipole_moment_integrals_ao = scf_reference_state.mol.intor("int1e_r", aosym="s1")
            return dipole_moment_integrals_ao
    elif vlx_success:
        molecule = scf_reference_state.task.molecule
        basis = scf_reference_state.task.ao_basis
        dipmom = vlx.compute_electric_dipole_integrals(molecule, basis)
        dipole_moment_integrals_ao = np.array(dipmom)
        return dipole_moment_integrals_ao
    elif pyscf_success:
        dipole_moment_integrals_ao = scf_reference_state.mol.intor("int1e_r", aosym="s1")
        return dipole_moment_integrals_ao
    else:
        raise ValueError("Neither VeloxChem nor PyScf could be loaded.") 

def select_theory_level(adc_level_valence, adc_level_core):
    """ Select the correct theory level to construct the
        state-to-state transition density matrices.
    """
    if "adc0" in adc_level_valence:
        if "adc0" in adc_level_core:
            return "adc0"
        else:
            raise ValueError("Incompatible ADC orders.", adc_level_valence, adc_level_core)
    if "adc1" in adc_level_valence:
        if "adc1" in adc_level_core:
            return "adc1"
        else:
            raise ValueError("Incompatible ADC orders.", adc_level_valence, adc_level_core)
    elif "adc2" in adc_level_valence:
        if "adc2" or "adc3" in adc_level_core:
            return "adc2"
        else:
            raise ValueError("Incompatible ADC orders.", adc_level_valence, adc_level_core)
    elif "adc3" in adc_level_valence:
        if "adc2" or "adc3" in adc_level_core:
            return "adc3"
        else:
            raise ValueError("Incompatible ADC orders.", adc_level_valence, adc_level_core)

def transform_to_nm(eV):
    """ Transforms from eV to nm.

        :param eV: the spectrum in eV.
        :returns: the spectrum in nm.
    """
    c = physical_constants['speed of light in vacuum'][0]
    h_in_eVs = physical_constants['Planck constant in eV s'][0]
    m_to_nm = 1e9
    nm = h_in_eVs * c * m_to_nm / eV 
    return nm

def read_from_h5(filename, label="rsp"):
    """ Reads the data from a checkpoint file and returns it as a dictionary.

        :param filename: the name of the checkpoint file.
        :param label   : the label for which data is read from file;
                         rsp - response; scf - SCF; opt - optimization;
                         vib - vibrational analysis.
    """
    res_dict = {}
    h5f = h5py.File(filename, "r")

    if label is None:
        h5f_dict = h5f
    else:
        h5f_dict = h5f[label]

    for key in h5f_dict:
        data = np.array(h5f_dict[key])
        res_dict[key] = data
    h5f.close()
    
    return res_dict

def add_broadening(bge, bgi, line_param=0.1, line_profile="lorentzian",
                   step=0.1, interval=None):
    """ Adds a Gaussian or Lorentzian broadening to a bar graph spectrum.

        :param bge         : the numpy array of energies.
        :param bgi         : the numpy array of intensities.
        :param line_param  : the line parameter.
        :param line_profile: the line profile (guassian or lrentzian)
        :param step        : the step size.
        :param interval    : the energy interval where the broadening
                             should be applied.
    """
    if interval is None:
        x_min = np.min(bge) - 5
        x_max = np.max(bge) + 5
    else:
        x_min = interval[0]
        x_max = interval[-1]

    x = np.arange(x_min, x_max, step)
    y = np.zeros((len(x)))
    
    # go through the frames and calculate the spectrum for each frame
    for xp in range(len(x)):
        for e, f in zip(bge, bgi):
            if line_profile in ['Gaussian', 'gaussian', "Gauss", "gauss"]:
                y[xp] += f * np.exp(-(
                    (e - x[xp]) / line_param)**2)
            elif line_profile in ['Lorentzian', 'lorentzian',
                                  'Lorentz', 'lorentz']:
                y[xp] += 0.5 * line_param * f / (np.pi * (
                        (x[xp] - e)**2 + 0.25 * line_param**2))
    return x, y


def get_maximum_intensity(bge, bgi, interval=None, atol=0.1):
    """ Returns the maximum intensity and corresponding energy
        position in an energy interval.

        :param bge     : the numpy array of energies.
        :param bgi     : the numpy array of intensities. 
        :param interval: the energy interval.
    """
    if interval is None:
        max_int = np.max(bgi)
        index = np.where(bgi == max_int)[0][0]
        en_max_int = bge[index]
    else:
        start = np.where(np.isclose(bge, interval[0], atol=atol))[0][0] 
        stop = np.where(np.isclose(bge, interval[1], atol=atol))[0][0]
        max_int = np.max(bgi[start:stop])
        index = np.where(bgi == max_int)[0][0]
        en_max_int = bge[index]

    return en_max_int, max_int, index

def get_symbols(xyz_file_name):
    """ Returns a list of unique atomic symbols from an xyz file.

        :param xyz_file_name: the file name
    """
    xyz_file = open(xyz_file_name, "r")

    symbols = []

    for line in xyz_file:
        parts = line.split()
        if parts:
            if len(parts) >= 4:
                symbols.append(parts[0])

    xyz_file.close()

    return symbols

def build_molecule(xyz_file_name, basis, ecp=None, symmetry=False,
                   charge=0, multiplicity=1):
    """ Returns the pyscf molecule object for a specific xyz file and basis set.

        :param xyz_file_name  : the file name.
        :param basis          : a basis set label (string),
                                or a dictionary defining the basis set
                                for each atom.
        :param ecp            : an ECP label (string),
                                or dictionary defining the ecp for each atom.
        :param symmetry_on    : if to use symmetry
    """
    pyscf_molecule = pyscf.gto.Mole()
    pyscf_molecule.atom = xyz_file_name
    pyscf_molecule.basis = basis
    pyscf_molecule.charge = charge
    pyscf_molecule.multiplicity = multiplicity

    if ecp is not None:
        pyscf_molecule.ecp = ecp
    pyscf_molecule.symmetry = symmetry

    pyscf_molecule.build()

    return pyscf_molecule

def run_scf_ground_state(pyscf_molecule, xc=None, restricted=True,
                        conv_tol=1e-8, max_cycles=150, verbose=0):
    """ Calculate the ground state reference.

        :param pyscf_molecule: the pyscf molecule object
        :param xc            : the exchange-correlation functional label (string)
                               if None -> run HF
        :param restricted    : if restricted.
        :param conv_tol      : the convergence threshold.
        :param max_cycles    : the maximum number of SCF cycles.
    """

    if xc is None:
        if restricted:
            scf = pyscf.scf.RHF(pyscf_molecule)
        else:
            scf = pyscf.scf.UHF(pyscf_molecule)
    else:
        if restricted:
            scf = pyscf.scf.RKS(pyscf_molecule)
        else:
            scf = pyscf.scf.UKS(pyscf_molecule)
        scf.xc = xc
    scf.conv_tol = conv_tol
    scf.max_cycle = max_cycles
    scf.verbose = verbose
    scf.kernel()

    return scf

def calculate_ionization_energy(pyscf_molecule, mo_index=0,
                                xc=None, conv_tol=1e-8, max_cycles=150, spin=1,
                                verbose=0):
    """ Calculates the ionization energy for the MO with index
        mo_index (0=core, -1=HOMO).

        :param pyscf_molecule: the pyscf molecule object.
        :param mo_index      : the index of the MO to be ionized.
        :param xc            : the exchange-correlation functional label (string)
                               if None -> run HF
        :param conv_tol      : the convergence threshold.
        :param max_cycles    : the maximum number of SCF cycles.
        :param spin          : 0 for hole in alpha, 1 for hole in beta.

        :return:
            a dictionary of the ionization energy, GS, and FCH reference states.
    """

    scf_gs_unrest = run_scf_ground_state(pyscf_molecule, xc=xc, restricted=False,
                                         conv_tol=conv_tol,
                                         max_cycles=max_cycles,
                                         verbose=verbose)
    mo_coeff_gs = copy.deepcopy(scf_gs_unrest.mo_coeff)
    occ = copy.deepcopy(scf_gs_unrest.mo_occ)
    if mo_index == -1:
        # find HOMO
        nocc = np.sum(scf_gs_unrest.mo_occ[spin,:]==1)
        mo_index = nocc - 1
    occ[spin][mo_index] = 0.0

    if xc is None:
        scf_ion = pyscf.scf.UHF(pyscf_molecule)
    else:
        scf_ion = pyscf.scf.UKS(pyscf_molecule)
        scf_ion.xc = xc

    scf_ion.conv_tol = conv_tol
    scf_ion.max_cycle = max_cycles
    scf_ion.verbose = verbose

    pyscf.scf.addons.mom_occ(scf_ion, mo_coeff_gs, occ)
    
    scf_ion.kernel()

    ie_H = scf_ion.e_tot - scf_gs_unrest.e_tot
    ie_eV = ie_H * hartree_in_ev()

    return {
        'IE in H': ie_H,
        'IE in eV': ie_eV,
        'SCF GS': scf_gs_unrest,
        'SCF ion': scf_ion
    }

def vlx_calculate_ionization_energy(vlx_molecule, vlx_basis, mo_index=0, xc=None,
                                   conv_thresh=1e-6, max_iter=150):
    """ Calculate the ionization energy using VeloxChem.

        :param vlx_molecule  : the VeloxChem molecule object.
        :param vlx_basis     : the VeloxChem basis set object.
        :param mo_index      : the index of the MO to be ionized.
                               (0=core, -1=HOMO).
        :param xc            : the exchange-correlation functional label (string)
                               if None -> run HF.
    """

    # Perform SCF calculation
    scf_gs = vlx.ScfRestrictedDriver()
    scf_settings = {'conv_thresh':conv_thresh, 'max_iter':max_iter}
    if xc is not None:
        method_settings = {"xcfun": xc}
    else:
        method_settings = {}
    scf_gs.update_settings(scf_settings, method_settings)
    scf_gs.ostream.mute()
    scf_gs_results = scf_gs.compute(vlx_molecule, vlx_basis)

    
    # Calculate SCF of the ionized molecule

    # Copy molecule object
    vlx_molecule_ionized = copy.deepcopy(vlx_molecule)

    # Change charge and multiplicity
    vlx_molecule_ionized.set_charge(1)
    vlx_molecule_ionized.set_multiplicity(2)

    nocc = vlx_molecule.number_of_alpha_electrons()
    occ_a = list(np.arange(nocc))
    occ_b = list(np.delete(occ_a, mo_index))

    scf_ion = vlx.ScfUnrestrictedDriver()
    scf_ion.maximum_overlap(vlx_molecule_ionized, vlx_basis,
                            scf_gs.mol_orbs, occ_a, occ_b)
    scf_ion.update_settings(scf_settings, method_settings)
    scf_ion.ostream.mute()
    scf_ion_results = scf_ion.compute(vlx_molecule_ionized, vlx_basis)

    ie_H = scf_ion.get_scf_energy() - scf_gs.get_scf_energy()
    ie_eV = ie_H * hartree_in_ev()

    return {
        'IE in H': ie_H,
        'IE in eV': ie_eV,
        'SCF GS': scf_gs_results,
        'SCF ion': scf_ion_results,
    }

def absorption_by_E2_matrix_diagonalization(pyscf_molecule, scf_gs,
                                            tda=False, cvs_space=None,
                                            fxc=None):
    """ Computes the absorption spectrum using TDDFT.

        :param pyscf_molecule: the pyscf molecule object
        :param scf_gs        : the SCF reference state.
        :param tda           : if to use the Tamm-Dancoff approximation.
        :param cvs_space     : a list of the core orbital indices
                               (for the CVS approximation).
    """

    print(pyscf.__path__)
    if tda:
        tdscf_drv = pyscf.tdscf.TDA(scf_gs)
    else:
        tdscf_drv = pyscf.tdscf.TDDFT(scf_gs)

    try:
        print("User defined fxc:", fxc)
        A, B = tdscf_drv.get_ab(user_defined_fxc=fxc)
    except TypeError:
        print("Regular fxc!")
        A, B = tdscf_drv.get_ab()

    nocc = A.shape[0]
    nvir = A.shape[1]

    mo_occ = scf_gs.mo_coeff[:, :nocc]
    mo_vir = scf_gs.mo_coeff[:, nocc:]
    nao = mo_occ.shape[0]

    electric_dipole_integrals_ao = np.sqrt(2) * pyscf_molecule.intor("int1e_r",
                                                                    aosym='s1')
    
    if tda:
        if cvs_space is None:
            E2 = A.reshape(nocc*nvir, nocc*nvir)
            S2 = np.identity(nocc * nvir)
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ,
                    electric_dipole_integrals_ao, mo_vir).reshape(3, nocc*nvir)
        else:
            ncore = len(cvs_space)
            eI = A[cvs_space, :, :, :]
            E2 = eI[:,:,cvs_space,:].reshape(ncore*nvir, ncore*nvir)
            S2 = np.identity(ncore * nvir)
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ[:,cvs_space],
                electric_dipole_integrals_ao, mo_vir).reshape(3, ncore*nvir)
        prop_grad = mu_mo
        omega, x = np.linalg.eigh(E2)

    else:
        if cvs_space is None:
            E2 = np.zeros((2*nocc*nvir, 2*nocc*nvir))
            E2[:nocc*nvir, :nocc*nvir] = A.reshape(nocc*nvir, nocc*nvir)
            E2[nocc*nvir:, nocc*nvir:] = A.reshape(nocc*nvir, nocc*nvir)
            E2[:nocc*nvir, nocc*nvir:] = -B.reshape(nocc*nvir, nocc*nvir)
            E2[nocc*nvir:, :nocc*nvir] = -B.reshape(nocc*nvir, nocc*nvir)
            S2 = np.identity(2 * nocc * nvir)
            S2[nocc*nvir:, nocc*nvir:] *= -1
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ,
                    electric_dipole_integrals_ao, mo_vir).reshape(3, nocc*nvir)
            prop_grad = np.zeros((3, 2*nocc*nvir))
            prop_grad[:, :nocc*nvir] = mu_mo
            prop_grad[:, nocc*nvir:] = -mu_mo
        else:
            ncore = len(cvs_space)
            E2 = np.zeros((2*ncore*nvir, 2*ncore*nvir))
            AI = A[cvs_space, :, :, :]
            BI = B[cvs_space, :, :, :]
            E2[:ncore*nvir, :ncore*nvir] = AI[:, :, cvs_space, :].reshape(
                                                         ncore*nvir, ncore*nvir)
            E2[ncore*nvir:, ncore*nvir:] = AI[:, :, cvs_space, :].reshape(
                                                        ncore*nvir, ncore*nvir)
            E2[:ncore*nvir, ncore*nvir:] = -BI[:, :, cvs_space, :].reshape(
                                                        ncore*nvir, ncore*nvir)
            E2[ncore*nvir:, :ncore*nvir] = -BI[:, :, cvs_space, :].reshape(
                                                        ncore*nvir, ncore*nvir)
            S2 = np.identity(2 * ncore * nvir)
            S2[ncore*nvir:, ncore*nvir:] *= -1
            mu_mo = np.einsum("mi,xmn,na->xia", mo_occ[:,cvs_space],
                    electric_dipole_integrals_ao, mo_vir).reshape(3, ncore*nvir)
            prop_grad = np.zeros((3, 2*ncore*nvir))
            prop_grad[:, :ncore*nvir] = mu_mo
            prop_grad[:, ncore*nvir:] = -mu_mo

        eigs, X = np.linalg.eig(np.matmul(np.linalg.inv(S2), E2))
        idx = np.argsort(eigs)
        omega = np.array(eigs)[idx]
        x = np.array(X)[:, idx]
        
        if cvs_space is None:
            omega = omega[nocc*nvir:]
            x = x[:, nocc*nvir:]
        else:
            omega = omega[ncore*nvir:]
            x = x[:, ncore*nvir:]

    n = omega.shape[0]
    tdms = np.zeros((n, 3))
    osc = np.zeros_like(omega)
    for k in range(n):
        Xf = x[:, k]
        Xf = Xf / np.sqrt(np.matmul(Xf.T, np.matmul(S2, Xf)))
        tdms[k] = np.einsum("i,xi->x", Xf, prop_grad)
        osc[k] = 2.0/3.0 * omega[k] * ( tdms[k,0]**2 
                                      + tdms[k,1]**2
                                      + tdms[k,2]**2)
            
    return {'eigenvalues': omega,
            'eigenvectors': x,
            'oscillator strengths': osc,
            'transition dipole moments': tdms,
            'E2': E2,
            'S2': S2}

def absortion_by_matvec(pyscf_molecule, scf_gs, nstates=5,
                        tda=False, cvs_space=None):
    """ NOT IMPLEMENTED!
        Computes the absorption spectrum using TDDFT.

        :param pyscf_molecule: the pyscf molecule object
        :param scf_gs        : the SCF reference state.
        :param nstates       : the number of excited states (roots).
        :param tda           : if to use the Tamm-Dancoff approximation.
        :param cvs_space     : a list of the core orbital indices
                               (for the CVS approximation).
    """
    raise NotImplementedError("Not implemented. Use absorption_spectrum_tdscf instead.")

def emission_spectrum_tdscf(scf_ch, nstates=5, tda=True):
    """ Computes the emission spectrum using TDDFT starting from
        a CH initial state.

        :param scf_ch        : the SCF reference state with a CH.
        :param nstates       : the number of transitions (roots).
        :param tda           : if to use the Tamm-Dancoff approximation.
    """

    if tda:
        tdscf_drv = pyscf.tdscf.TDA(scf_ch)
    else:
        raise ValueError("Only TDA is possible.")

    tdscf_drv.nstates = nstates

    tdscf_drv.positive_eig_threshold = -1e10
    tdscf_drv.kernel()

    return tdscf_drv

def absorption_spectrum_tdscf(scf_gs, nstates=5, cvs_space=None, tda=False):
    """ Computes the absorption spectrum using TDDFT.

        WARNING: The CVS approximation only works with
                 a restricted reference state.

        :param scf_gs   : the SCF reference state (for CVS, restricted only).
        :param nstates  : the number of transitions (roots).
        :param tda      : if to use the Tamm-Dancoff approximation.
        :param cvs_space: a list of core orbital indices
                          for the CVS approximation.
    """

    if tda:
        tdscf_drv = pyscf.tdscf.TDA(scf_gs)
    else:
        tdscf_drv = pyscf.tdscf.TDDFT(scf_gs)

    tdscf_drv.nstates = nstates
    tdscf_drv.cvs_space = cvs_space
    tdscf_drv.kernel()

    return tdscf_drv

def check_minimum_denominator(scf_ref, warning_thr=0.1):
    """ Calculates the minimum denominator of the t amplitudes.
        1/(ei+ej-ea-eb)

        :param scf_ref    : the scf reference state (unrestricted.
        :param warning_thr: the warning threshold blow which the denominator
                            is considered too small.
    """

    if scf_ref.mo_energy.shape[0] == 1:
        raise ValueError("Use an unrestricted reference state.")

    mo_energies_alpha = scf_ref.mo_energy[0]
    mo_energies_beta = scf_ref.mo_energy[1]
    mo_occ_a = scf_ref.mo_occ[0]
    mo_occ_b = scf_ref.mo_occ[1]
    
    occ_a = np.where(mo_occ_a == 1)[0]
    occ_b = np.where(mo_occ_b == 1)[0]
    virt_a = np.where(mo_occ_a == 0)[0]
    virt_b = np.where(mo_occ_b == 0)[0]
    
    e_occ_a = mo_energies_alpha[occ_a]
    e_occ_b = mo_energies_beta[occ_b]
    e_vir_a = mo_energies_alpha[virt_a]
    e_vir_b = mo_energies_alpha[virt_b]

    e_vv_aa = e_vir_a + e_vir_a.reshape(-1, 1) # epsilon_a + epsilon_b (as 2D matrix)
    e_vv_ab = e_vir_a + e_vir_b.reshape(-1, 1)
    e_vv_bb = e_vir_b + e_vir_b.reshape(-1, 1)

    min_denominator = 1e9
    where = None
    
    for ei in e_occ_a:
        for ej in e_occ_a:
            e_aaaa = (e_vv_aa - ei - ej)
            min_ij = np.abs(np.min(e_aaaa))
            if min_ij < min_denominator:
                min_denominator = min_ij
                where = np.where(np.abs(e_aaaa)==min_ij)
                ea = e_vir_a[where[0][0]]
                eb = e_vir_a[where[1][0]]
                min_denom_data = {'spin': 'aaaa', 'mo energies': [ei, ej, ea, eb]}
                
    for ei in e_occ_a:
        for ej in e_occ_b:
            e_abab = (e_vv_ab - ei - ej)
            min_ij = np.abs(np.min(e_abab))
            if min_ij < min_denominator:
                min_denominator = min_ij
                where = np.where(np.abs(e_abab)==min_ij)
                ea = e_vir_a[where[0][0]]
                eb = e_vir_b[where[1][0]]
                min_denom_data = {'spin': 'abab', 'mo energies': [ei, ej, ea, eb]}

    for ei in e_occ_b:
        for ej in e_occ_b:
            e_bbbb = (e_vv_bb - ei - ej)
            min_ij = np.abs(np.min(e_bbbb))
            if min_ij < min_denominator:
                min_denominator = min_ij   
                where = np.where(np.abs(e_bbbb)==min_ij)
                ea = e_vir_b[where[0][0]]
                eb = e_vir_b[where[1][0]]
                min_denom_data = {'spin': 'bbbb', 'mo energies': [ei, ej, ea, eb]}
        
    if min_denominator > warning_thr:
        return {'ok': True, 'min_denominantor': min_denominator, 'min_denom_data': min_denom_data} 
    else:
        return {'ok': False, 'min_denominantor': min_denominator, 'min_denom_data': min_denom_data}

def absorption_spectrum_adc(scf_gs, adc_level="adc2",
                            nsinglets=None, cvs_space=None,
                            nstates=None, ntriplets=None, nspinflip=None,
                            frozen_core=None, frozen_virtual=None,
                            conv_tol=1e-8):
    """ Computes the absorption spectrum using ADC.

        :param scf_gs        : the SCF reference state.
        :param adc_level     : the ADC theory level.
        :param nsinglets     : the number of singlet excited states (roots).
        :param cvs_space     : a list of the core orbital indices
                               (for the CVS approximation).
        :param nstates       : the number of states.
        :param ntriplets     : the number of triplets.
        :param nspinflip     : the number of spin-flip states.
        :param frozen_core   : a list/index for the occupied orbitals which should be frozen.
        :param frozen_virtual: a list/index for the virtual orbitals which should be frozen.

        :return:
            the adc result object.
        
    """
    if cvs_space is not None:
        if "cvs" not in adc_level:
            adc_level = "cvs-" + adc_level
    adc = adcc.run_adc(scf_gs, method=adc_level, 
                       core_orbitals=cvs_space, n_states=nstates,
                        n_singlets=nsinglets, n_triplets=ntriplets,
                        n_spin_flip=nspinflip, conv_tol=conv_tol,
                        frozen_core=frozen_core, frozen_virtual=frozen_virtual)
    return adc

def emission_spectrum_adc(scf_ch, adc_level="adc2", min_denominator=0.1,
                          nsinglets=None, nstates=None, ntriplets=None,
                          nspinflip=None,
                          conv_tol=1e-8):
    """ Computes the absorption spectrum using ADC.

        :param scf_gs        : the SCF reference state with a CH.
        :param adc_level     : the ADC theory level.
        :param nsinglets     : the number of singlet excited states (roots).
        :param nstates       : the number of states.
        :param ntriplets     : the number of triplets.
        
    """
    min_denominator_results = check_minimum_denominator(scf_ch, min_denominator)
    
    if min_denominator_results['ok']:
        adc = adcc.run_adc(scf_ch, method=adc_level, 
                   n_states=nstates, n_singlets=nsinglets, n_triplets=ntriplets,
                   n_spin_flip=nspinflip, conv_tol=conv_tol)
        return adc
    else:
        warning_text = "Warning!: minimum denominator below "
        warning_text += "the threshold %.2e" % min_denominator
        print(warning_text)
        return min_denominator_results

def calculate_ionization_energy_mp(adc_abs, adc_emis, mp_level=2):
    """ Calculates the ionization energy using MP from two
        previous ADC calculations.

        :param adc_abs      : the ADC absorption results.
        :param adc_emis     : the ADC emission results.
        :param mp_level     : int, the MP level.
    """

    tot_en_gs = adc_abs.reference_state.energy_scf
    tot_en_ion = adc_emis.reference_state.energy_scf

    corrections = {} 
    for mp in range(2, mp_level+1):
        mp_correction_gs = adc_abs.ground_state.energy_correction(mp)
        mp_correction_ion = adc_emis.ground_state.energy_correction(mp)
        tot_en_gs += mp_correction_gs
        tot_en_ion += mp_correction_ion
        level = "MP%d" % mp
        corrections[level] = (mp_correction_gs, mp_correction_ion)
        
    ie_H = tot_en_ion - tot_en_gs
    ie_eV = ie_H * hartree_in_ev()

    return {
        'IE in H': ie_H,
        'IE in eV': ie_eV,
        'MP corrections': corrections
    }

def perform_iad_analysis(data, name, window=15, line_param=0.5,
                        line_step=0.05, line_shape="lorentzian",
                        ref_method="adc2x", max_int_stop=5,
                        save=False, figsize=(16, 12), 
                        subplots=(3, 4), cmap=None, plot_diff_spectra=True,
                        tick_step=2):
    """ Performs the IAD analysis for a specific molecule and edge.

        :param data        : the DataFrame with all molecules.
        :param name        : the name of the molecule
                             and edge (e.g. C01: Carbon edge, molecule 01).
        :param window      : the energy window where line broadening
                             will be applied (in eV).
        :param line_param  : the line parameter.
        :param line_step   : the step size used when broadening
                             the bar graph spectrum.
        :param line_shape  : the line shape (gaussian or lorentzian).
        :param ref_method  : the name of the method used as a reference.
        :param max_int_stop: the index of the last position where to
                             look for the maximum peak (to which the
                             spectra will be aligned).
        :param save        : flag to save the plots as svg figures.
        :param figsize     : the size parameters for the plot.
        :param subplots    : the subplots; tuple containing the number of
                             rows and number of columns.
        :param cmap        : the color map.
        :param plot_diff_spectra: flag to plot the difference spectra.
        :param tick_step   : the step-size for the x-axis ticks.

        :returns:
            a DataFrame containig the shifted and braodened spectrum,
            difference spectrum, shift, intensity ratio,
            area ratio, and IAD.
    """

    # Prepare reference data
    try:
        ref_en = data[name][ref_method][0]
        ref_osc = data[name][ref_method][1]
    except TypeError:
        error_text = "** Error: Could not read the reference data"
        raise TypeError(error_text, name, ref_method)

    if np.min(ref_en) + window - 2 < np.max(ref_en):
        energy_window = (np.min(ref_en) - 2, np.min(ref_en) + window - 2)
    else:
        warning_text = "** Warning: Requested energy window "
        warning_text += "larger than maximum calculated energy. Readjusting."
        print(warning_text)
        energy_window = (np.min(ref_en) - 2, np.max(ref_en))
    
    xticks = np.arange(np.round(energy_window[0]), np.round(energy_window[-1]),
                       tick_step)
    ref_x, ref_y = add_broadening(ref_en, ref_osc,
                                  line_param=line_param,
                                  step=line_step,
                                  line_profile=line_shape,
                                  interval=energy_window)
    ref_area = scipy.integrate.simpson(ref_y, x=ref_x)

    max_en, max_int, ref_index = get_maximum_intensity(ref_en,
                            ref_osc, interval=(ref_en[0], ref_en[max_int_stop]))

    # Prepare result dictionary and figure settings
    if cmap is None:
        cmap = cmaps['viridis'] # default color map
    
    molecule_dict = {}
    fig = plt.figure(figsize=figsize)
    how_many = len(data[name].keys())
    color = np.linspace(0, 0.8, how_many)

    # Calculate parameters and plot spectra
    i = 0
    for method in data[name].keys():
        method_dict = {}
        try:
            en = data[name][method][0]
            osc = data[name][method][1]

            # Calculate parameters:
            max_en, max_int, index = get_maximum_intensity(en, osc,
                                             interval=(en[0], en[max_int_stop]))
            shift = ref_en[ref_index] - en[index]
            i_ratio = osc[index] / ref_osc[ref_index]
            x, y = add_broadening(en + shift, osc, line_param=line_param,
                                 step=line_step, line_profile=line_shape,
                                 interval=energy_window)
            area = scipy.integrate.simpson(y, x=x)
            area_ratio = area / ref_area
            diff_y = ref_y / ref_area - y / area
            iad = scipy.integrate.simpson(np.abs(diff_y), x=x)
            
            method_dict['area-norm spectrum'] = (x, y/area)
            method_dict['area-norm diff'] = diff_y
            method_dict['shift'] = shift
            method_dict['I ratio'] = i_ratio
            method_dict['area_ratio'] = area_ratio
            method_dict['iad'] = iad
            txt =  "shift: %7.2f\nIAD: %7.2f\n" % (shift, iad)
            txt += "I/I0: %7.2f\nA/A0: %7.2f" % (i_ratio, area_ratio)
        except:
            x = 0
            y = 0
            area = 1
            txt =  "shift: NA, IAD : NA\n"
            txt += "I/I0 : NA, A/A0: NA"

        plt.subplot(subplots[0], subplots[1], i+1)
        plt.plot(ref_x, ref_y/ref_area, color="gray", linewidth=1)
        plt.fill_between(ref_x, ref_y/ref_area, step="pre", alpha=0.4,
                        color="gray", label=ref_method)
        plt.plot(x, y/area, label=method, color=cmap(color[i]), linewidth=2)
        plt.text(energy_window[1]-0.5, 0.55, txt, fontsize=9,
                horizontalalignment="right")
    
        if i % subplots[1] == 0:
            plt.ylabel("Osc. str. (area norm.)")
        plt.xlabel("Photon energy (eV)")
        plt.axis(xmin=np.round(energy_window[0]),
                 xmax=np.round(energy_window[1]), ymin=-0.01, ymax=1)
        plt.xticks(xticks)
        plt.legend()
        molecule_dict[method] = method_dict
        i += 1
    if save:
        plt.savefig(name + "_area_norm_spectra.svg")
    plt.show()

    molecule_data = pd.DataFrame(molecule_dict)

    # Plot the difference spectra
    if plot_diff_spectra:
        fig = plt.figure(figsize=figsize)
        i = 0
        for method in molecule_data.columns:
            try:
                x = molecule_data[method]["area-norm spectrum"][0]
                diff_y = molecule_data[method]["area-norm diff"]
                IAD = molecule_data[method]["iad"]
                txt = "IAD: %6.2f" % IAD
            except:
                x = 0
                diff_y = 0
                txt = "IAD: NA"
        
            plt.subplot(subplots[0], subplots[1], i+1)
            plt.plot(x, diff_y, label=method, color=cmap(color[i]), linewidth=2)
            plt.axhline(y=0, color="black", ls="--")
            plt.text(energy_window[1]-0.5, 0.55, txt, fontsize=10,
                     horizontalalignment="right")
        
            if i % subplots[1] == 0:
                plt.ylabel("Area-norm diff.")
            plt.xlabel("Photon energy (eV)")
            plt.axis(xmin=np.round(energy_window[0]),
                     xmax=np.round(energy_window[1]), ymin=-1, ymax=1)
            plt.xticks(xticks)
            plt.legend()
            i += 1
        if save:
            plt.savefig(name + "_area_norm_diff_spectra.svg")
        plt.show()

    return molecule_data

def calculate_rixs_spectrum(scf_reference_state, adc_valence_results, adc_core_results,
                            core_orbitals, omega, theta=0, gamma=0.004557,
                            include_omega_product=True, valence_interval=None):
    """ Calculates the RIXS spectrum using the two-step approach at the ADC theory level.

        :param scf_reference_state:
            The SCF reference state (from PyScf or VeloxChem).
        :param adc_valence_results:
            The ADC ExcitedStates object obtained for the valence states from an adcc calculation.
        :param core_states:
            The ADC ExcitedStates object obtained for the core states from an adcc calculation using CVS.
        :param core_orbitals:
            The list of core orbitals used in the CVS ADC calculation.
        :param omega:
            The angular frequency corresponding to the incoming photon (in H).
        :param theta:
            The angle between the polarization of the incoming photon and the scattering direction (in rad!)
        :param gamma:
            The full-width at half maximum energy broadening of the intermediate (core) states,
            corresponding to the inverse lifetime.
        :param valence_interval:
            A tuple of two indices indicating which intrerval of valence states should be considered.
            (0, 10) will consider the first 10 valence-excited states (10, 20) will consider 
            excited states 10-19.

        :returns:
            A dictionary of the outgoing angular frequencies (a.u.), outgoing photon energies (eV),
            energy loss (eV), RIXS cross-sections (a.u.), and RIXS scattering tensors (a.u.).
    """
    for_total = time.time()
    start = time.time()
    if valence_interval is None:
        valence_states = adc_valence_results.excitation_vector
        valence_excitation_energies = adc_valence_results.excitation_energy
        istart = 0
        istop = len(valence_excitation_energies)
    else:
        istart = valence_interval[0]
        istop = valence_interval[1]
        valence_states = adc_valence_results.excitation_vector[istart:istop]
        valence_excitation_energies = adc_valence_results.excitation_energy[istart:istop]

    core_states = adc_core_results.excitation_vector
    core_excitation_energies = adc_core_results.excitation_energy

    theory_level = select_theory_level(adc_valence_results.method.name,
                                       adc_core_results.method.name)

    dipole_moment_integrals = get_dipole_moment_integrals(scf_reference_state)

    # Elastic peak
    gs_to_core_tdms = list(adc_core_results.transition_dipole_moment)
    rixs_tensor = calculate_rixs_scattering_matrix(gs_to_core_tdms, gs_to_core_tdms,
                                                   core_excitation_energies, 0,
                                                   omega=omega, gamma=gamma,
                                                   include_omega_product=include_omega_product)
    sigma = calculate_rixs_transition_strength(rixs_tensor, w=omega, wf0=0, theta=theta)

    # Save elastic peak results
    omega_prime = [omega]
    rixs_scattering_tensors = [rixs_tensor]
    rixs_scattering_cross_sections = [sigma]
    # For the elastic peak, the core-to-valence tdms are the same as gs-to-core
    all_core_to_valence_tdms = [gs_to_core_tdms] 


    print("Pre-requisites and elastic peak...: %.2f s." % (time.time() - start))

    # Inelastic peaks
    for vi, valence_excited in enumerate(valence_states):
        omega_prime.append(omega - valence_excitation_energies[vi])
        core_to_valence_tdms = []
        # Compute the core-to-valence transition dipole moments
        for ci, core_excited in enumerate(core_states): 
            opdm = construct_transition_density_matrix(scf_reference_state=scf_reference_state,
                                                       adc_excitation_vector_1=valence_excited,
                                                       adc_excitation_vector_2=core_excited,
                                                       theory_level=theory_level,
                                                       cvs_space=core_orbitals)
            tdm = calculate_transition_dipole_moment(transition_density_matrix=opdm,
                                                     dipole_moment_integrals=dipole_moment_integrals)
            core_to_valence_tdms.append(tdm)

        print("S%d 1PTDMs...: %.2f s." % (istart + vi + 1, time.time() - start))
        start = time.time()

        rixs_tensor = calculate_rixs_scattering_matrix(core_to_valence_tdms, gs_to_core_tdms,
                                                       core_excitation_energies,
                                                       valence_excitation_energies[vi],
                                                       omega=omega, gamma=gamma,
                                                       include_omega_product=include_omega_product)

        print("S%d Scattering tensor...: %.2f s." % (istart + vi + 1, time.time() - start))
        start = time.time()

        sigma = calculate_rixs_transition_strength(rixs_tensor, w=omega,
                                                   wf0=valence_excitation_energies[vi],
                                                   theta=theta)
        print("S%d Cross-section..: %.2f s." % (istart + vi + 1, time.time() - start))
        start = time.time()

        rixs_scattering_tensors.append(rixs_tensor)
        rixs_scattering_cross_sections.append(sigma)
        all_core_to_valence_tdms.append(core_to_valence_tdms)

    outgoing_photon_energies_ev = np.array(omega_prime) * hartree_in_ev()
    energy_loss_ev = omega * hartree_in_ev() - outgoing_photon_energies_ev

    print("Total: %.2f" % (time.time() - for_total))

    return {
        'omega_prime': omega_prime,
        'outgoing_photon_energies_ev': outgoing_photon_energies_ev,
        'energy_loss_ev': energy_loss_ev,
        'rixs_scattering_cross_sections': np.array(rixs_scattering_cross_sections),
        'rixs_scattering_tensors': rixs_scattering_tensors,
        'gs_to_core_transition_dipole_moments': np.array(gs_to_core_tdms),
        'core_to_valence_transition_dipole_moments': np.array(all_core_to_valence_tdms),
    }

def calculate_rixs_scattering_matrix(core_to_valence_tdms, gs_to_core_tdms,
                                     core_excitation_energies, valence_excitation_energy,
                                     omega, gamma=0.004557, include_omega_product=True):
    """ Calculates the RIXS scattering matrix for a given frequency and valence-excited state.

        :param core_to_valence_tdms:
            The list of core-to-valence transition dipole moments.
        :param gs_to_core_tdms:
            The list of ground state to core-excited state transition dipole moments.
        :param core_excitation_energies:
            A list of core-excitation energies (in Hartree).
        :param valence_excitation_energy:
            The valence excitation energy (in Hartree).
        :param omega:
            The energy/frequency of the incident photon (a.u.).
        :param gamma:
            The life-time broadening parameter (FWHM). In the scattering matrix equation,
            the half-width half maximum (HWHM) is used (1/2 gamma)

        :returns:
            A 3x3 numpy array of the RIXS scattering amplitudes.
    """

    scattering_matrix = np.zeros((3,3), dtype='complex128')

    # FWHM -> HWHM for RIXS
    gamma_hwhm = gamma * 0.5
    for i, ctov_tdm in enumerate(core_to_valence_tdms):
        omega_product = (  core_excitation_energies[i] 
                         - valence_excitation_energy ) * core_excitation_energies[i]
        omega_factor = 1.0 / (core_excitation_energies[i] - omega - 1j * gamma_hwhm)
        gtoc_tdm = gs_to_core_tdms[i]
        if include_omega_product:
            scattering_matrix += omega_product * omega_factor * np.einsum("x,y->xy", ctov_tdm, gtoc_tdm)
        else:
            # For debugging
            scattering_matrix += omega_factor * np.einsum("x,y->xy", ctov_tdm, gtoc_tdm)

    return scattering_matrix


def calculate_rixs_transition_strength(rixs_scattering_matrix, w, wf0,
                                       theta=0.0):
    """ Calculates the RIXS transition strength sigma at a given
        angle theta between the polarization of incoming photon 
        and the scattered photon direction.

        References: 
            - Daniel R. Nascimento et al. 
              https://doi.org/10.1021/acs.jctc.1c00144
            - Dirk R. Rehn et al.
              https://pubs.acs.org/doi/10.1021/acs.jctc.7b00636

        :param rixs_scattering_matrix: the 3 x 3 scattering matrix
                                       for a particular final 
                                       valence-excited state f.
        :param w: the angular frequency of the incoming photon
                  (in a.u. equal to the incoming photon energy;
                   corresponds to the selected core-excitation energy).
                  (in a.u.)
        :param wf0: the valence-excitation energy corresponding
                    to the selected valence-excited state. (in a.u.)
        :param theta: the angle between the polarization of the
                      incoming photon and the scattering direction (in rad!).

        returns:
            the transition strength in a.u.
    """
    shape = rixs_scattering_matrix.shape
    w_prime = w - wf0
    if len(shape)==3:
        einsum_string = ["xAB,xAB->x", "xAB,xBA->x", "xAA,xBB->x"]
    else:
        einsum_string = ["AB,AB->", "AB,BA->", "AA,BB->"]
    sigma = w_prime / w * 1/15.0 * (
              (2 - 0.5 * (np.sin(theta))**2)
            * np.einsum(einsum_string[0], rixs_scattering_matrix,
                                   rixs_scattering_matrix.conjugate())
            + (0.75 * (np.sin(theta))**2 - 0.5) 
            * (  np.einsum(einsum_string[1], rixs_scattering_matrix,
                    rixs_scattering_matrix.conjugate()) 
            + np.einsum(einsum_string[2], rixs_scattering_matrix,
                        rixs_scattering_matrix.conjugate())
              )
                                    )
    return sigma

def construct_transition_density_matrix(scf_reference_state, adc_excitation_vector_1,
                                        adc_excitation_vector_2=None,
                                        theory_level=None, cvs_space=None):
    """ Constructs the one-particle transition density matrix between two ADC excited states.

        :param scf_reference_state:
            the PyScf or VeloxChem SCF reference state.
        :param adc_excitation_vector_1:
            the excitation vector of the first selected excited state.
        :param adc_excitation_vector_2:
            the excitation vector of the second selected excited state.
            if not None, the state-to-state transition density matrix will be calculated.
            if None, the ground state to excited state transition density matrix will be calculated
        :param theory_level:
            the theory level.
        :param cvs_space:
            the indices of the core orbitals (in the case of a CVS calculation).

        returns:
            the transition density matrix in AO basis.

    """

    if theory_level is None:
        raise ValueError("Please specify the ADC order.")

    vector_1_ph = adc_excitation_vector_1.ph.to_ndarray()

    if adc_excitation_vector_2 is None:
        tdm = from_gs_adc0_tdm(scf_reference_state, vector_1_ph)
        if "adc1" in theory_level or "adc2" in theory_level or "adc3" in theory_level:
            tdm += from_gs_adc1_tdm(scf_reference_state, vector_1_ph)
        if "adc2" in theory_level or "adc3" in theory_level:
            vector_1_pphh = adc_excitation_vector_1.pphh.to_ndarray()
            tdm += from_gs_adc2_tdm(scf_reference_state, vector_1_ph, vector_1_pphh)  
        return tdm
    else:
        nocc, mo_coeff = get_nocc_mo_coeff(scf_reference_state)
        # ADC(0) and ADC(1) have the same expression for the state-to-state 1PTDM
        vector_1_ph, vector_2_ph = get_excitation_vectors_singles_block(adc_excitation_vector_1,
                                                                        adc_excitation_vector_2,
                                                                        nocc,
                                                                        cvs_space)
        tdm = state_to_state_adc0_tdm(scf_reference_state, vector_1_ph, vector_2_ph)
        # ADC(2), ADC(2)-x, and ADC(3) have the same expression for the state-to-state 1PTDM
        if "adc2" in theory_level or "adc3" in theory_level:
            vector_1_pphh, vector_2_pphh = get_excitation_vectors_doubles_block(adc_excitation_vector_1,
                                                                                adc_excitation_vector_2,
                                                                                nocc,
                                                                                cvs_space)
            tdm += state_to_state_adc2_tdm(scf_reference_state, vector_1_ph, vector_2_ph,
                                           vector_1_pphh, vector_2_pphh)
        return tdm

def get_excitation_vectors_singles_block(vector_1, vector_2, nocc=None, cvs_space=None):
    """ Takes as an input two adc excitation vectors and returns two singles
        blocks as numpy arrays of the same dimensions. In the case in which one
        of the vectors is from a CVS calculation, adds zeros in the non-core
        occupied block.

        :param vector_1:
            The ADC vector 1.
        :param vector_2:
            The ADC vector 2.
        :param cvs_space:
            The list of core orbital indices from the CVS calculation (if applicable).
    """
    raw_vector_1 = vector_1.ph.to_ndarray()
    raw_vector_2 = vector_2.ph.to_ndarray()
    n1 = raw_vector_1.shape[0]
    n2 = raw_vector_2.shape[0]

    if n1 == n2:
        return raw_vector_1, raw_vector_2
    else:
        if cvs_space is None:
            raise ValueError("Please define the CVS space.")
        else:
            alpha_beta = cvs_space.copy()
            for c in cvs_space:
                alpha_beta.append(c+nocc)
            if n1 > n2:
                new_vector_2 = np.zeros_like(raw_vector_1)
                new_vector_2[alpha_beta] = raw_vector_2
                return raw_vector_1, new_vector_2
            else:
                for c in cvs_space:
                    alpha_beta.append(c+n2)
                new_vector_1 = np.zeros_like(raw_vector_2)
                new_vector_1[alpha_beta] = raw_vector_1
                return new_vector_1, raw_vector_2

def get_excitation_vectors_doubles_block(vector_1, vector_2, nocc=None, cvs_space=None):
    """ Takes as an input two adc excitation vectors and returns two doubles 
        blocks as numpy arrays of the same dimensions. In the case in which one
        of the vectors is from a CVS calculation, adds zeros in the non-core
        occupied block.

        :param vector_1:
            The ADC vector 1.
        :param vector_2:
            The ADC vector 2.
        :param cvs_space:
            The list of core orbital indices from the CVS calculation (if applicable).
    """
    raw_vector_1 = vector_1.pphh.to_ndarray()
    raw_vector_2 = vector_2.pphh.to_ndarray()
    n1 = raw_vector_1.shape[0]
    n2 = raw_vector_2.shape[0]

    if n1 == n2:
        return raw_vector_1, raw_vector_2
    else:
        if cvs_space is None:
            raise ValueError("Please define the CVS space.")
        else:
            core = cvs_space.copy()
            for c in cvs_space:
                core.append(c+nocc)
            occupied = list(set(np.arange(0, 2*nocc, 1)) - set(core))
            if n1 > n2:
                new_vector_2 = np.zeros_like(raw_vector_1)
                for i, core_id in enumerate(core):
                    new_vector_2[occupied, core_id, :, :] = raw_vector_2[:, i, :, :]
                return raw_vector_1, new_vector_2
            else:
                new_vector_1 = np.zeros_like(raw_vector_2)
                for i, core_id in enumerate(core):
                    new_vector_1[occupied, core_id, :, :] = raw_vector_1[:, i, :, :]
                return new_vector_1, raw_vector_2


def state_to_state_adc0_tdm(scf_reference_state, vector_1_ph, vector_2_ph):
    """ Calculates the ADC(0) and ADC(1) state-to-state one-particle
        transition density matrix in AO basis.

        :param scf_reference_state:
            The PyScf or VeloxChem SCF reference state.
        :param vector_1_ph:
            The singles block of the first excitation vector (numpy array).
        :param vector_2_ph:
            The singles block of the second excitation vector (numpy array).
    """
    nocc, mo_coeff = get_nocc_mo_coeff(scf_reference_state)
    mo_occ = mo_coeff[:, :nocc]
    mo_vir = mo_coeff[:, nocc:]
    nvir = mo_vir.shape[1]

    # tdm_vv = np.einsum('ib,ia->ab', vector_1_ph, vector_2_ph)
    # tdm_oo = -np.einsum('ja,ia->ij', vector_1_ph, vector_2_ph)
    tdm_vv = np.matmul(vector_1_ph.T, vector_2_ph)
    tdm_oo = -np.matmul(vector_1_ph, vector_2_ph.T).T

    # the factor 2.0 is for alpha + beta
    tdm_ao = 2.0 * (  np.linalg.multi_dot([mo_occ, tdm_oo[:nocc, :nocc], mo_occ.T])
                    + np.linalg.multi_dot([mo_vir, tdm_vv[:nvir, :nvir], mo_vir.T])
                   )
    return tdm_ao

def state_to_state_adc2_tdm(scf_reference_state, vector_1_ph, vector_2_ph, vector_1_pphh, vector_2_pphh):
    """ Calculates the ADC(2), ADC(2)-x and ADC(3/2) state-to-state one-particle
        transition density matrix in AO basis.

        :param scf_reference_state:
            The PyScf or VeloxChem SCF reference state.
        :param vector_1_ph:
            The singles block of the first excitation vector (numpy array).
        :param vector_2_ph:
            The singles block of the second excitation vector (numpy array).
    """
    nocc, mo_coeff = get_nocc_mo_coeff(scf_reference_state)
    mo_occ = mo_coeff[:, :nocc]
    mo_vir = mo_coeff[:, nocc:]
    nvir = mo_vir.shape[1]

    adc_reference_state = adcc.ReferenceState(scf_reference_state)
    lazymp = adcc.LazyMp(adc_reference_state)
    ooov = adc_reference_state.eri("o1o1o1v1").to_ndarray()
    ovvv = adc_reference_state.eri("o1v1v1v1").to_ndarray()
    
    t2 = lazymp.t2("o1o1v1v1").to_ndarray()
    
    bov = lazymp.df("o1v1").to_ndarray()

    rho_oo = -0.5 * np.einsum("ikab,jkab->ij", t2, t2)
    rho_vv = 0.5 *  np.einsum("ijac,ijbc->ab", t2, t2)
    rho_ov = -0.5 * (   np.einsum("ijbc,jabc->ia", t2, ovvv)
                      + np.einsum("jkib,jkab->ia", ooov, t2)
                       ) / bov
    
    r1 = np.einsum("jb,ijab->ia", vector_1_ph, t2)
    r2 = np.einsum("jb,ijab->ia", vector_2_ph, t2)
    
    # dm0_vv = np.einsum('ib,ia->ab', vector_1_ph, vector_2_ph)
    # dm0_oo = -np.einsum('ia,ja->ij', vector_1_ph, vector_2_ph)
    dm0_vv = np.matmul(vector_1_ph.T, vector_2_ph)
    dm0_oo = -np.matmul(vector_1_ph, vector_2_ph.T)

    dm0_ov = -2.0 * np.einsum("jb,ijab->ia", vector_1_ph, vector_2_pphh)
    dm0_vo = -2.0 * np.einsum("ijab,jb->ai", vector_1_pphh, vector_2_ph)
    
    vv_t2dmvv = np.einsum("klbc,cd->klbd", t2, dm0_vv)
    vv_t2dmoo = np.einsum("klbc,jk->ljbc", t2, dm0_oo)
    vv_t2r1 = np.einsum("ikac,kc->ia", t2, r1)
    vv_t2r2 = np.einsum("ikbc,kc->ib", t2, r2)

    tdm_vv = (
                 2.0 * np.einsum('ijac,ijbc->ab', vector_1_pphh, vector_2_pphh)
               # - 0.5 * np.einsum("ac,cb->ab", dm0_vv, rho_vv)
               # - 0.5 * np.einsum("ac,cb->ab", rho_vv, dm0_vv)
               - 0.5 * np.matmul(dm0_vv, rho_vv)
               - 0.5 * np.matmul(rho_vv, dm0_vv)
               - 0.5 * np.einsum("klad,klbd->ab", t2, vv_t2dmvv)
               + 1.0 * np.einsum("ljbc,jlac->ab", vv_t2dmoo, t2)
               # + 0.5 * np.einsum("ia,ib->ab", vv_t2r1, vector_2_ph)
               # + 0.5 * np.einsum("ia,ib->ab", vector_1_ph, vv_t2r2)
               # + 1.0 * np.einsum("ia,ib->ab", r2, r1)
               + 0.5 * np.matmul(vv_t2r1.T, vector_2_ph)
               + 0.5 * np.matmul(vector_1_ph.T, vv_t2r2)
               + 1.0 * np.matmul(r2.T, r1)
            )

    oo_dmoot2 = np.einsum("lk,jlcd->jkcd", dm0_oo, t2)
    oo_t2dmvv = np.einsum("ikcd,db->ikcb", t2, dm0_vv)
    oo_t2r1 = np.einsum("jkac,kc->ja", t2, r1)
    oo_t2r2 = np.einsum("ikac,kc->ia", t2, r2)

    tdm_oo = ( 
               - 2.0 * np.einsum('ikab,jkab->ij', vector_1_pphh, vector_2_pphh)
               # + 0.5 * np.einsum("ik,kj->ij", dm0_oo, rho_oo)
               # + 0.5 * np.einsum("ik,kj->ij", rho_oo, dm0_oo)
               + 0.5 * np.matmul(dm0_oo, rho_oo)
               + 0.5 * np.matmul(rho_oo, dm0_oo)
               - 0.5 * np.einsum("ikcd,jkcd->ij", t2, oo_dmoot2)
               + 1.0 * np.einsum("ikcb,jkcb->ij", oo_t2dmvv, t2)
               # - 0.5 * np.einsum("ia,ja->ij", vector_2_ph, oo_t2r1)
               # - 0.5 * np.einsum("ia,ja->ij", oo_t2r2, vector_1_ph)
               # - 1.0 * np.einsum("ia,ja->ij", r1, r2)
               - 0.5 * np.matmul(vector_2_ph, oo_t2r1.T)
               - 0.5 * np.matmul(oo_t2r2, vector_1_ph.T)
               - 1.0 * np.matmul(r1, r2.T)
              )

    ov_t2v1vv = np.einsum("klca,klcb->ab", t2, vector_1_pphh)
    ov_t2v1oo = np.einsum("ikcd,jkcd->ij", t2, vector_1_pphh)

    tdm_ov = dm0_ov + 1.0 * ( 
             - np.einsum("ijab,bj->ia", t2, dm0_vo)
             # - np.einsum("ib,ba->ia", rho_ov, dm0_vv)
             # + np.einsum("ij,ja->ia", dm0_oo, rho_ov)
             # - np.einsum("ib,ab->ia", vector_2_ph, ov_t2v1vv)
             # - np.einsum("ij,ja->ia", ov_t2v1oo, vector_2_ph)
             - np.matmul(rho_ov, dm0_vv)
             + np.matmul(dm0_oo, rho_ov)
             - np.matmul(vector_2_ph, ov_t2v1vv.T)
             - np.matmul(ov_t2v1oo, vector_2_ph)
            )

    vo_t2v2vv = np.einsum("klca,klcb->ab", t2, vector_2_pphh)
    vo_t2v2oo = np.einsum("ikcd,jkcd->ij", t2, vector_2_pphh)
   
    tdm_vo =  dm0_vo + 1.0 * (
             - np.einsum("ijab,jb->ai", t2, dm0_ov)
             # - np.einsum("ib,ab->ai", rho_ov, dm0_vv)
             # + np.einsum("ji,ja->ai", dm0_oo, rho_ov)
             # - np.einsum("ib,ab->ai", vector_1_ph, vo_t2v2vv)
             # - np.einsum("ij,ja->ai", vo_t2v2oo, vector_1_ph)
             - np.matmul(dm0_vv, rho_ov.T)
             + np.matmul(rho_ov.T, dm0_oo.T)
             - np.matmul(vo_t2v2vv, vector_1_ph.T, )
             - np.matmul(vector_1_ph.T, vo_t2v2oo.T)
            )

    # The factor 2 accounts for alpha + beta
    tdm_ao = 2.0 * (  np.linalg.multi_dot([mo_occ, tdm_oo[:nocc, :nocc], mo_occ.T])
              + np.linalg.multi_dot([mo_vir, tdm_vv[:nvir,:nvir], mo_vir.T])
              + np.linalg.multi_dot([mo_occ, tdm_ov[:nocc, :nvir], mo_vir.T])
              + np.linalg.multi_dot([mo_vir, tdm_vo[:nvir, :nocc], mo_occ.T])
            )

    return tdm_ao

def from_gs_adc0_tdm(scf_reference_state, vector_ph):
    """ Calculates the ADC(0) one-particle transition density matrix contribution
        from the GS to an excited state with excitation vector vector_ph.

        :param scf_reference_state:
            The PyScf or VeloxChem SCF reference state.
        :param vector_ph:
            The singles block of the excitation vector (numpy array).

        returns:
            The ADC(0) transition density matrix contribution in AO basis.
    """
    nocc, mo_coeff = get_nocc_mo_coeff(scf_reference_state)
    mo_occ = mo_coeff[:, :nocc]
    mo_vir = mo_coeff[:, nocc:]
    nvir = mo_vir.shape[1]

    tdm_ao = (  np.linalg.multi_dot([mo_vir, vector_ph[:nocc, :nvir].T, mo_occ.T])
              + np.linalg.multi_dot([mo_vir, vector_ph[nocc:, nvir:].T, mo_occ.T])
             )

    return tdm_ao

def from_gs_adc1_tdm(scf_reference_state, vector_ph):
    """ Calculates the ADC(1) one-particle transition density matrix contribution
        from the GS to an excited state with excitation vector vector_ph.

        :param scf_reference_state:
            The PyScf or VeloxChem SCF reference state.
        :param vector_ph:
            The singles block of the excitation vector (numpy array).

        returns:
            The ADC(1) transition density matrix contribution in AO basis.
    """
    nocc, mo_coeff = get_nocc_mo_coeff(scf_reference_state)
    mo_occ = mo_coeff[:, :nocc]
    mo_vir = mo_coeff[:, nocc:]

    nvir = mo_vir.shape[1]

    adc_ref_state = adcc.ReferenceState(scf_reference_state)
    t2 = adcc.LazyMp(adc_ref_state).t2("o1o1v1v1").to_ndarray()
    tdm_ov = -np.einsum('ijab,jb->ia', t2, vector_ph, optimize=True)
    tdm_ao = ( np.linalg.multi_dot([mo_occ, tdm_ov[:nocc, :nvir], mo_vir.T])
             + np.linalg.multi_dot([mo_occ, tdm_ov[nocc:, nvir:], mo_vir.T])
             )

    return tdm_ao

def from_gs_adc2_tdm(scf_reference_state, vector_ph, vector_pphh):
    """ Calculates the ADC(2) one-particle transition density matrix contribution
        from the GS to an excited state with excitation vector vector_ph.

        :param scf_reference_state:
            The PyScf or VeloxChem SCF reference state.
        :param vector_ph:
            The singles block of the excitation vector (numpy array).

        returns:
            The ADC(2) transition density matrix contribution in AO basis.
    """
    nocc, mo_coeff = get_nocc_mo_coeff(scf_reference_state)
    mo_occ = mo_coeff[:, :nocc]
    mo_vir = mo_coeff[:, nocc:]
    nvir = mo_vir.shape[1]

    adc_ref_state = adcc.ReferenceState(scf_reference_state)

    lazymp = adcc.LazyMp(adc_ref_state)
    t2 = adcc.LazyMp(adc_ref_state).t2("o1o1v1v1").to_ndarray()
    foo = adc_ref_state.fock("o1o1").to_ndarray()
    fvv = adc_ref_state.fock("v1v1").to_ndarray()
    e_occ = np.diagonal(foo).copy()
    e_vir = np.diagonal(fvv).copy()
    e_ov = e_occ.reshape(-1,1) - e_vir
    e_vv = e_vir.reshape(-1,1) + e_vir
    e_oo = e_occ.reshape(-1,1) + e_occ
    e_oovv = ( - e_oo.reshape((2*nocc, 2*nocc, 1, 1)) 
              + e_vv.reshape((1, 1, 2*nvir, 2*nvir))
             )
    vvvv = adc_ref_state.eri("v1v1v1v1").to_ndarray()
    ovov = adc_ref_state.eri("o1v1o1v1").to_ndarray()
    oooo = adc_ref_state.eri("o1o1o1o1").to_ndarray()
    ovvv = adc_ref_state.eri("o1v1v1v1").to_ndarray()
    oovv = adc_ref_state.eri("o1o1v1v1").to_ndarray()
    ooov = adc_ref_state.eri("o1o1o1v1").to_ndarray()

    rov =  0.5 * (
            + np.einsum("ijbc,jabc->ia", t2, ovvv)
            + np.einsum("jkib,jkab->ia", ooov, t2)
        ) / e_ov

    # TODO: pre-calculate some of these terms...
    tdm_vo = (   0.50 * np.einsum("ijab,jkbc,kc->ai", t2, t2, vector_ph)
               - 0.25 * np.einsum("kjac,kjbc,ib->ai", t2, t2, vector_ph)
               - 0.25 * np.einsum("ja,ikbc,jkbc->ai", vector_ph, t2, t2)
               )

    tdm_ov = (    np.einsum("kjbc,jaib,ikac,ia->kc", t2, ovov, 1/e_oovv, vector_ph)
                + np.einsum("jiab,jckb,ikac,ia->kc", t2, ovov, 1/e_oovv, vector_ph)
                + np.einsum("jicb,jakc,ikab,ia->kb", t2, ovov, 1/e_oovv, vector_ph)
                + np.einsum("kjac,jbic,ikab,ia->kb", t2, ovov, 1/e_oovv, vector_ph)
               - 0.50 * np.einsum("jicd,abcd,ijab,ia->jb", t2, vvvv, 1/e_oovv, vector_ph)
               + 0.50 * np.einsum("klab,klij,ijab,ia->jb", t2, oooo, 1/e_oovv, vector_ph)
              )

    tdm_oo =  ( - np.einsum("ia,ja->ij", rov, vector_ph)
                - np.einsum("ikab,jkab->ji", vector_pphh, t2)
              )
    tdm_vv = ( np.einsum("ia,ib->ab", vector_ph, rov)
             + np.einsum("ijac,ijbc->ab", vector_pphh, t2)
                )
    # alpha + beta
    tdm_ao = 2 * (  np.linalg.multi_dot([mo_occ, tdm_oo[:nocc, :nocc], mo_occ.T])
                  + np.linalg.multi_dot([mo_vir, tdm_vv[:nvir, :nvir], mo_vir.T])
                  + np.linalg.multi_dot([mo_vir, tdm_vo[:nvir, :nocc], mo_occ.T])
                  + np.linalg.multi_dot([mo_occ, tdm_ov[:nocc, :nvir], mo_vir.T])
             )

    return tdm_ao

def calculate_transition_dipole_moment(transition_density_matrix, dipole_moment_integrals):
    """ Calculates the transition dipole moment.

        :param transition_density_matrix: the one-particle transition density matrix in AO basis.
        :param dipole_moment_integrals: a numpy array of the dipole moment integrals in AO basis.

        returns the transition dipole moment in a.u.
    """
    return -np.einsum("mn,xmn->x", transition_density_matrix, dipole_moment_integrals)

def calculate_oscillator_strength(transition_dipole_moment, delta_e):
    """ Calculates the osccillator strength.

        :param transition_dipole_moment: the transition dipole moment.
        :param delta_e: the transition energy in a.u.

        returns the dimensionless oscilaltor strength.
    """

    return 2.0/3.0 * delta_e * np.linalg.norm(transition_dipole_moment)**2
