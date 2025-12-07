import numpy as np
import tenpy
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks import site
from tenpy.simulations.measurement import m_energy_MPO
import os
import sys 
import pickle
tenpy.tools.misc.setup_logging(to_stdout="INFO")

"""
Python script to reproduce the iDMRG data in the article "Charge-4e Superconductivity in a Hubbard model" (arXiv:2312.13348)
by Martina O. Soldini, Mark H. Fischer, and Titus Neupert.
Affiliation: Physics Institute, University of Zurich, Winterthurerstrasse 190, 8057 Zurich, Switzerland

This script defines the charge-4e Hubbard model studied throughout the paper, as well as the charge-2e Hubbard model,
in the classes "charge4e_model" and "charge2e_model" respectively.
The "run" function runs the iDMRG algorithm for the 4e or 2e model, and computes several observables. Correlations
are obtained with the "correlations_4emodel" and "correlations_2emodel" functions. The particle number variance is
computed with the "particle_number_variance" function. The "run" function returns a dictionary with the results of the
iDMRG simulation, including the computed observables, the state obtained, and information on the iDMRG run.

This script runs the iDMRG algorithm for the 4e model, and the results if the iDMRG calculations are stored in a 
dictionary as saved as .pkl files. 
"""

class charge4e_model(CouplingMPOModel):
    """
        Define the charge-4e Hubbard model as in Eq. (3) of arXiv:2312.13348
        
        H = \sum_i H_i - t \sum_{i, l, s} c^{\dag}_{i, l, s} c{\dag}_{i+1, l, s} + h.c.
        
        H_i = -mu \sum_{}l, s} n_{i, l, s} 
                + U \sum_{l} n_{i, l, up} n_{i, l, down} 
                    + V \sum_{l, s} n_{i, 1, s} n_{i, 2, s}

        where i is the site index, l=1, 2 the orbital index, s=up,down the spin index, 
        c^{\dag}_{i, l, s} and  c{\dag}_{i, l, s} the electronic creation and annihilation operator
        of the state at site i, orbital l and spin s. n_{i, l, s} = c^{\dag}_{i, l, s} c_{i, l, s} is the 
        density operator, and n_{i, l} = \sum_s n_{i, l, s}.

        The model parameters are:
            - (-mu): the chemical potential, fixed to -1.
            - t: the nearest-neighbour hopping amplitude
            - U: the Hubbard interaction within the same orbital
            - V: the Hubbard interaction between different orbitals
        The parameters U and V, with -mu, define delta = -2mu + U + 2V, which is the parameter describing the 
        energy of the four-fold occupied state in a single (decoupled) site.
        Additional mean-field supercondunducting perturbation (to compute the susceptibility) can be added
        (as in Eqs. (11) of arXiv:2312.13348):
            - Del2: the superconducting mean-field perturbation in the same orbital
            - Del4: the superconducting mean-field perturbation between different orbitals

        The model uses the "Ladder" lattice default type in Tenpy.
        The model is defined on a ladder,  with two spinful fermionic orbitals per rung, structured 
        as in the following schematics

        ---- 1, i-1 ---- 1, i ---- 1, i+1 ---- 
                |          |          |  
        ---- 2, i-1 ---- 2, i ---- 2, i+1 ----

        Sites are numbered according to the order: [ (1, i-1), (2, i-1), (1, i), (2, i), (1, i+1), (2, i+1), ...]

    """
    default_lattice = "Ladder"
    force_default_lattice = True
   
    def init_sites(self, model_params):
        """Define the type of sites in the unit cell: two spinful fermionic sites per unit cell"""
        cons_N = model_params.get('cons_N', "None")
        cons_Sz = model_params.get('cons_Sz', "None")
        ferm1 = site.SpinHalfFermionSite(cons_N = cons_N, cons_Sz = cons_Sz)
        ferm2 = site.SpinHalfFermionSite(cons_N = cons_N, cons_Sz = cons_Sz)
        sites = [ferm1, ferm2]
        site.set_common_charges(sites, new_charges='independent')
        return [ferm1, ferm2]
    
    def init_terms(self, model_params):
        #Get the parameters for the model
        t = model_params.get('t', 0.)
        mu = model_params.get('mu', -1.)
        U = model_params.get('U', 0.)
        V = model_params.get('V', 0.)

        for i in [0, 1]:
            #Nearest-neighbour Hopping
            self.add_coupling(-t, i, 'Cdd', i, 'Cd', [1], plus_hc=True)
            self.add_coupling(-t, i, 'Cdu', i, 'Cu', [1], plus_hc=True)
            #Chemical potential
            self.add_onsite(-mu, i, 'Nd')
            self.add_onsite(-mu, i, 'Nu')
            #Hubbard U within same orbitals
            self.add_onsite(U, i, 'NuNd')

        #Hubbard V between different orbitals
        self.add_coupling(V, 0, 'Nu', 1, 'Nu', [0])
        self.add_coupling(V, 0, 'Nu', 1, 'Nd', [0])
        self.add_coupling(V, 0, 'Nd', 1, 'Nu', [0])
        self.add_coupling(V, 0, 'Nd', 1, 'Nd', [0])
    
        #Superconducting mean-field perturbation (as in Eqs. (11) of arXiv:2312.13348)
        Del2 = model_params.get('Del2', 0.)
        Del4 = model_params.get('Del4', 0.)
        
        if np.abs(Del2)>0:
            for i in [0, 1]:
                self.add_onsite(Del2, i, "Cu Cd", plus_hc=True)
        if np.abs(Del4)>0:
            self.add_coupling(Del4, 0, "Cu Cd", 1, "Cu Cd", [0])
            self.add_coupling(Del4, 0, "Cdu Cdd", 1, "Cdu Cdd", [0])

class charge2e_model(CouplingMPOModel):
    """
        Define the charge-2e Hubbard model as in Eq. (A1) of arXiv:2312.13348, based on the "Chain" 
        lattice default type in Tenpy. At each site there is a spinful fermionic orbital.
    
        ---- i-1 ---- i ---- i+1 ---- 
    """
    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        """Define the type of sites in the unit cell (two per unit cell)"""
        cons_N = model_params.get('cons_N', "None")
        cons_Sz = model_params.get('cons_Sz', "None")
        ferm = site.SpinHalfFermionSite(cons_N = cons_N, cons_Sz = cons_Sz)
        sites = ferm
        return ferm
    
    def init_terms(self, model_params):
        #Get the parameters for the model
        #Parameters for the charge-2e model
        t = model_params.get('t', 1.)
        mu = model_params.get('mu', 0.)
        U = model_params.get('U', 0.)

        #Nearest neighbour hopping
        self.add_coupling(-t, 0, 'Cdd', 0, 'Cd', [1], plus_hc=True)
        self.add_coupling(-t, 0, 'Cdu', 0, 'Cu', [1], plus_hc=True)
        #Chemical potential
        self.add_onsite(mu, 0, 'Nd')
        self.add_onsite(mu, 0, 'Nu')
        #Hubbard U within same orbital
        self.add_onsite(U, 0, 'NuNd')

def correlations_4emodel(psi):
    """
    Correlations in the charge-4e model

    This function evaluates the one-, two-, and four-particle correlations in the charge-4e model, 
    over a range of distances, corresponding to idx_list.

    We define the creation and annihilation operators of the electronic state at site j, with orbital
    l = 0, 1 and spin s = up, down as c_{j, l, s}, and c^dag_{j, l, s}, respectively.
    The correlations are defined as follows:
        1) One particle correlations:
            corr1[s, |i-j|] = <c_{i, l, s} c^dag_{j, l, s}> 
            with s: 0 -> spin-up, s: 1 -> spin-down.
        2) Two particle correlations:
            corr2[S+Orb, |i-j|] = <c_{i, 1, up} c_{i, 1+Orb, down+S} c^dag_{j, 1, up} c^dag_{j, 1+Orb, down+S}> 
            with total orbital index Orb = 0, 1, and total spin index S = 0, 1.
            Note that O = 1 and S=1 is not allowed by Pauli exclusion principle.
        3) Four particle correlations:
            corr4[|i-j|] = <c_{i, 1, up} c_{i, 1, down} c_{i, 2, up} c_{i, 2, down} x
                                c^dag_{i, 1, up} c^dag_{i, 1, down} c^dag_{i, 2, up} c^dag_{i, 2, down} >.
    """

    #We define the reference point (idx0) and range (idx_list) to compute correlation functions
    idx0 = 0 
    idx_list = np.arange(2, 100, 2) #Jump by two as there are two orbitals in each unit cell

    #One-particle correlations:
    corr1 = np.zeros((2, len(idx_list)+1))
    corr1[0, 0] = psi.expectation_value_term([("Nu", idx0)])
    corr1[1, 0] = psi.expectation_value_term([("Nd", idx0)])
    corr1[0, 1:] = psi.term_correlation_function_right([("Cu", 0)], [("Cdu", 0)], i_L = idx0, j_R = idx_list).flatten()
    corr1[1, 1:] = psi.term_correlation_function_right([("Cd", 0)], [("Cdd", 0)], i_L = idx0, j_R = idx_list).flatten()
    
    #Two-particle correlations:
    corr2 = np.zeros((3, len(idx_list)+1))
    corr2[0, 0] = psi.expectation_value_term([("NuNd", idx0)])
    for j, N in zip([1, 2], ["Nd", "Nu"]):
        corr2[j, 0] = psi.term_correlation_function_right([("Nu", 0)], [(N, 1)], i_L = idx0, j_R = [idx0]).flatten()
    for orb in range(2):
        corr2[orb, 1:] = psi.term_correlation_function_right([("Cu", 0), ("Cd", orb)],\
                                        [("Cdu", 0), ("Cdd", orb)], i_L = idx0, j_R = idx_list).flatten()
    corr2[2, 1:] = psi.term_correlation_function_right([("Cu", 0), ("Cu", orb)],\
                                        [("Cdu", 0), ("Cdu", orb)], i_L = idx0, j_R = idx_list).flatten()
    
    #Four-particle correlations:
    corr4 = psi.term_correlation_function_right([("Cu", 0), ("Cd", 0), ("Cu", 1), ("Cd", 1)],\
                                     [("Cdu", 0), ("Cdd", 0), ("Cdu", 1), ("Cdd", 1)], i_L = idx0, j_R = idx_list).flatten()
    corr4_0 = psi.term_correlation_function_right([("NuNd", 0)], [("NuNd", 1)], i_L = idx0, j_R = [idx0]).flatten()
    corr4 = np.concatenate((corr4_0, corr4))

    return corr1, corr2, corr4, idx_list

def correlations_2emodel(psi):
    """
    Correlations in the charge-2e model

    This function evaluates the one-, two-particle correlations in the charge-2e model, 
    over a range of distances, corresponding to idx_list.

    We define the creation and annihilation operators of the electronic state at site j, and 
    spin s = up, down as c_{j, s}, and c^dag_{j, s}, respectively.
    The correlations are defined as follows:
        1) One particle correlations:
            corr1[s, |i-j|] = <c_{i, s} c^dag_{j, s}> 
            with s: 0 -> spin-up, s: 1 -> spin-down.
        2) Two particle correlations:
            corr2[|i-j|] = <c_{i, up} c_{i, down+S} c^dag_{j, up} c^dag_{j, down+S}> 
            with total orbital index Orb = 0, 1, and total spin index S = 0, 1.
            Note that O = 1 and S=1 is not allowed by Pauli exclusion principle.
    """
    #We define the reference point (idx0) and range (idx_list) to compute correlation functions
    idx0 = 0 
    idx_list = np.arange(1, 100, 1) #Only one site per unit cell

    #One-particle correlations:
    #We compute both spin-up and spin-down correlations as a sanity check.
    corr1 = np.zeros((2, len(idx_list)+1))
    corr1[0, 0] = psi.expectation_value_term([("Nu", idx0)])
    corr1[1, 0] = psi.expectation_value_term([("Nd", idx0)])
    corr1[0, 1:] = psi.term_correlation_function_right([("Cu", 0)], [("Cdu", 0)], i_L = idx0, j_R = idx_list).flatten()
    corr1[1, 1:] = psi.term_correlation_function_right([("Cd", 0)], [("Cdd", 0)], i_L = idx0, j_R = idx_list).flatten()
    
    #Two-particle correlations:
    corr2 = np.zeros(len(idx_list)+1)
    corr2[0] = psi.expectation_value_term([("NuNd", idx0)])
    corr2[1:] = psi.term_correlation_function_right([("Cu", 0), ("Cd", 0)],\
                                                     [("Cdu", 0), ("Cdd", 0)], i_L = idx0, j_R = idx_list).flatten()
    return corr1, corr2, None, idx_list

def particle_number_variance(psi, densities):
    """
    This function returns the variance of the particle number operator
    evaluated on systems with finte size 2<l<lmax.

        First, the expectation values of the form <n_i n_j> for i,j=0,...,lmax
    are computed and stored.
    We store <n_i n_i> in the dictionary "onsite_NN", and <n_i n_j}> (j>i) in the dictionary 
    "apart_NN". The onsite_NN and apart_NN are dictionary have keys 0 and 1, indicating 
    the unit cell of reference (uc == i%2 in <n_i n_j>).
    We define: 
        n_i = n_{i, \ell=0} + n_{i, \ell=1} 
    where i is the site index and \ell is the orbital index. Therefore,
        <n_i n_j> = <n_{i, \ell=0} n_{j, \ell=0}>  
                        + <n_{i, \ell=1} n_{j, \ell=1}> 
                            + 2<n_{i, \ell=0} n_{j, \ell=1}> (*)
    Note that <n_i n_j> = <n_{i%2} n_{j-i+i%2}> due to translation symmetry,
    hence we only need to store <n_0 n_j> and <n_1 n_j> expectation values with
    0<=j<lmax and 1<=j<lmax, respectively.
    
        The expectation values stored in onsite_NN and apart_NN are used to construct
    the density squared operator (NN).
    Keeping in mind that there are two inequivalent sites in the two-site iDMRG 
    algorithm, we construct the <NN> operator on lmax sites as
    |n_0 n_0         n_0 n_1        n_0 n_2       ....        n_0 n_lmax|
    |n_1 n_0         n_1 n_1        n_1 n_2       ....        n_1 n_lmax|
    |n_2 n_0         n_2 n_1        n_2 n_2       ....        n_2 n_lmax|
    |                                 ....                              |
    |n_lmax n_0      n_lmax n_1     n_lmax n_2       ....  n_lmax n_lmax|
    where n_i is the density operator at site i, and n_{2k} and n_{2k'+1} 
    with k = 0,..., l//2-1, (and (l//2) if l is odd), and k'=0, ..., l//2-1 
    are the two sets of inequivalent sites.

        Finally, the variance of the particle number operator is computed for 
    system sizes l, with 2<l<lmax, and returned as a numpy array with the first 
    column containing l, and the second var(N_l)/l.
    """
    lmax = 500 # sets the summation cutoff in computing the particle number operator,
        # N = \sum_i=1^lmax N
        # and the squared particle number NN: NN = \sum_i=1^lmax \sum_j=1^lmax <n_i n_j> 
    
    onsite_NN = {0:{}, 1:{}} #keys: 0th and 1st unit cells
    apart_NN = {0:{}, 1:{}} #keys: 0th and 1st unit cells
    jlist = np.arange(2, lmax*2, 2)
    for uc in range(2):
        for i, j in zip([0, 0, 1], [0, 1, 1]):
            onsite_NN[uc][f"{i}{j}"] = psi.expectation_value_term([("Ntot", i+2*uc), ("Ntot", j+2*uc)]).flatten()
            apart_NN[uc][f"{i}{j}"] = psi.term_correlation_function_right([("Ntot", i)],\
                                             [("Ntot", j)], i_L = 0 + uc*2, j_R = jlist+uc*2).flatten()
    #Compute <n_i n_j> from <n_{i, ell} n_{j, ell'}> as in (*) above
    diagonal0 = onsite_NN[0]["00"] + onsite_NN[0]["11"] + 2*onsite_NN[0]["01"]
    diagonal1 = onsite_NN[1]["00"] + onsite_NN[1]["11"] + 2*onsite_NN[1]["01"]
    off_diagonal0 = apart_NN[0]["00"] + apart_NN[0]["11"] + 2*apart_NN[0]["01"]
    off_diagonal1 = apart_NN[1]["00"] + apart_NN[1]["11"] + 2*apart_NN[1]["01"]

    # Construct the <NN> operator:
    density_squared_matrix = np.zeros((lmax, lmax))
    for i in range(lmax):
        density_squared_matrix[i, i] = diagonal0 if i%2==0 else diagonal1
        for j in range(i):
            if i%2==0:
                density_squared_matrix[i, j] = off_diagonal0[i-j-1] #dd1[i-j-1]
            else:
                density_squared_matrix[i, j] = off_diagonal1[i-j-1] #dd2[i-j-1]
            density_squared_matrix[j, i] = density_squared_matrix[i, j]
    
    # Compute the variance on systems of size l=2, ..., lmax, divided by l.
    lvals = np.arange(2, lmax)

    #Compute <N_l^2>/l for l in lvals
    partial_sum_NN = [np.sum(density_squared_matrix[:l, :l])/l for l in lvals]
    #Compute <N_l>^2/l for l in lvals
    partial_sum_N2 = [(l/4)*np.sum(densities)**2 if l%2==0  else
                      ((l//2)**2 * np.sum(densities)**2 \
                            + np.sum(densities[:2])**2 \
                                + 2*(l//2)*np.sum(densities[:2])*np.sum(densities))/l 
                    for l in lvals]
    #Compute varN_l/l for l in lvals
    varianceNL = (np.array(partial_sum_NN) - np.array(partial_sum_N2))
    
    return np.stack((lvals, varianceNL))

def run(which_model, model_params, chi, init_state, L=2, max_err=1e-5, max_sweeps=1000):
    """ 
        Run iDMRG on the charge4e_model (if which_model="4e") or the charge_2e model  (if which_model="2e"), 
        with model parameters defined by the dictionary model_params, at bond dimension chi, for a two 
        site algorighm (L=2) and initial state defined by init_state.
        After the iDMRG terminates, this function computes some observables from the 
        state obtained (psi): the one-, two-, and four-particle correlations, energy, density 
        expectation value and variance, correlation length, and entanglement entropy.
        This function returns the result dictionary with all the computed observables, psi, and information on
        the iDMRG runs.
    """

    #List of increasing bond dimension to use across the iDMRG runs
    chi_list = {0:10, 11:chi//2, 21: int(3*chi/4), 51:chi}
    #Parameters of the dmrg algorithm
    dmrg_params = {
        'chi_list': chi_list, #between sweep 0-20: use chi/2, btw 20-50: use (3/4)chi, after 100: use chi.
        'trunc_params': {'chi_max':chi, 'svd_min': 1.e-10, 'trunc_cut': 1e-8},
        "lanczos_params":{"cutoff": 1.e-13}, #precision of the lanczos method
        'update_env': 10,
        'max_E_err': max_err, #precision in energy 
        'max_S_err': max_err, #precision in entropy
        'min_sweeps': 50,
        'max_sweeps': max_sweeps,
        'mixer': True,
        "mixer_params": {"amplitude": 1.e-5,
                         "decay":1.2,
                         "disable_after":30}
    }
    
    #Create dictionary to store the parameters and results of the iDMRG simulation
    results = dict(init_state=init_state, dmrg_params = dmrg_params, model_params = model_params)
    
    #Generate model with initial parameters
    model = charge4e_model(model_params) if which_model=="4e" else charge2e_model(model_params)

    #Initial state ansatz (particle-conservation breaking state)
    psi = MPS.from_product_state(model.lat.mps_sites(), init_state * L, model.lat.bc_MPS)
    
    assert L==2 #Check that the parameters are compatible with the two-site iDMRG algorithm
    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)

    print(' ---- Run iDMRG ---- ')
    engine.run() #Runs the iDMRG algorithm
    print(' ---- iDMRG finished ---- ')

    ###################* Calculate observables and store results #################   
    #Correlation functions 
    results['corr1'], results['corr2'], results['corr4'], results["idx_list"] = correlations_4emodel(psi=psi) if which_model=="4e" else correlations_2emodel(psi=psi)

    #Local density
    results['densities'] = psi.expectation_value(["Ntot"]).flatten()

    #Particle number variance
    if which_model=="4e":
        results['density_squared_matrix'] = particle_number_variance(psi=psi, densities=results['densities'])

    #Expectation value of P and Q operators
    results["Q"] = psi.expectation_value_term([("Cdd", 0), ("Cdu", 0), ("Cdd", 1), ("Cdu", 1)])
    results["P0_P1"] = [psi.expectation_value_term([("Cdd", 0), ("Cdu", j)]) for j in [0, 1]]

    #Correlation length
    results['corr_length'] = psi.correlation_length()
    
    #Energy of the ground state
    m_energy_MPO(results, psi, model, simulation = engine, results_key='energies')

    #Ground state tensor
    results['psi'] = psi

    #Convergence info on the iDMRG sweeps
    results["iDMRG_convergence"] = engine.sweep_stats

    return results

if __name__ == "__main__":
    #Running the iDMRG algorithm for the charge-4e model.
    #Parameters of the 4e model
    delta = float(sys.argv[1]) #Value of delta = 2|mu| + U + 2V 
    t = float(sys.argv[2]) #nearest neighbour hopping amplitude
    mu = -1. #chemical potential, fixed.
    r = float(sys.argv[3]) #ratio r:=V/U
    U = (delta - 2)/(2*r +1)
    V = r*U

    #iDMRG parameters
    L = 2 # two-site iDMRG
    chi = int(sys.argv[4]) #max bond dimension: min(cutoff, Hilbert space size)
    max_err = 1e-5

    #Initial state ansatz
    init_state = [np.array([1, 1, 1, 1])/2.] * 2 #Mixes different particle sectors.

    #Model parameters
    model_params = {"t":t, "delta":delta, "r":V/U, "U":U, "V":V, "L":L, "bc_MPS": "infinite"}

    filename = './iDMRG_charge4e/charge4eiDMRG_delta_{:.3f}_t_{:.3f}_r_{:.3f}_chi_{:d}_maxerr_{:.0e}'.format(delta, t, r, chi, max_err)
    if not os.path.exists(filename + '.pkl'):
        results = run(which_model="4e", model_params=model_params, chi=chi, init_state=init_state, max_err=max_err)
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(results, f)
