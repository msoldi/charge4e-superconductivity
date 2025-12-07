import numpy as np
import tenpy
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks import site
from tenpy.simulations.measurement import m_energy_MPO

import os, sys, pickle

tenpy.tools.misc.setup_logging(to_stdout="INFO")

"""
Python script for "Charge-4e Superconductivity in a Hubbard model" (arXiv:2312.13348)
by Martina O. Soldini, Mark H. Fischer, and Titus Neuper
Affiliation: Physics Institute, University of Zurich, Winterthurerstrasse 190, 8057 Zurich, Switzerland

This script launches the iDMRG calculations for the effective spin-1/2 model discussed in the article.
The effective spin-1/2 model is defined in the class "spin_chain_eff_model", and the appropriate
"run" function that runs the iDMRG algorithm on the spin effective model is also defined.
"""

class spin_chain_eff_model(CouplingMPOModel):
    """
        Spin chain as low energy effective model of the charge-4e model

        This class implements the spin effective model in a one-dimensional chain 
        with a spin-1/2 degree of freedom at every lattice site, with structure
        defined by the tenpy lattice Chain, and SpinHalfSite at every site. 
        The model Hamiltonian is
        
        H = \sum_i J(S^+_{i} S^-_{i+1} + S^-_{i} S^+_{i+1}) + JDelta \sum_i S^z_i S^z_{i+1} 
                + 2 delta \sum_i S^z_i + K \sum_i S^z_i S^z_{i+2}

        The effective model parameters J, JDelta, delta, and K are obtained from the
        hopping amplitude t, the chemical potential mu, and the interaction parameters U and V.
    """
    default_lattice = "Chain"
    force_default_lattice = True
   
    def init_sites(self, model_params):
        cons_Sz = model_params.get("cons_Sz", "None")
        spin = site.SpinHalfSite(conserve = cons_Sz)
        return spin
    
    def init_terms(self, model_params):
        #Get the parameters for the model
        t = model_params.get('t', 1.)
        mu = model_params.get('mu', -1.)
        U = model_params.get('U', 0.)
        V = model_params.get('V', 0.)

        #Define effective model parameters for the effective spin model (See App. B in arXiv:2312.13348)
        r = V/U
        W = -(2 + delta/mu)
        Zr = -((1+2*r) * (1+5*r))/(4 * r* (1+r))
        Yr = -(((1 - r + 2 * r**2))/(4 * r * (1 + r)))
        Xr = (2 * r * (3 + 2 * r))/((2 + 3*r) * (1 + 4*r))

        J = 2 * Zr * t**4/(mu**3)
        JDelta = -2*(W*(t**2/mu)  + (Yr+Xr) * t**4/(mu**3))
        K = (Xr + 1) * t**4/(mu**3)

        #Sx_i Sx_i+1 and analog for y, x terms
        self.add_coupling(J, 0, 'Sx', 0, 'Sx', [1])
        self.add_coupling(J, 0, 'Sy', 0, 'Sy', [1])
        self.add_coupling(JDelta, 0, 'Sz', 0, 'Sz', [1])
        
        #Magnetic field term
        self.add_onsite(2*delta, 0, 'Sz')

        #Next-to-nearest-neigbor interaction
        self.add_coupling(K, 0, 'Sz', 0, 'Sz', [2])


def run(chi, U, V, mu, t, delta, init_state, max_err):
    """
        Run the iDMRG algorithm on the effective spin-1/2 model
    """

    chi_list = {0: chi//2, 21: int(3*chi/4), 51:chi}
    #Parameters of the dmrg algorithm
    dmrg_params = {
        'chi_list': chi_list, #between sweep 0-30: use chi_max 20, btw 30-100: use chi_max 50, after 100: 100.
        'trunc_params': {'chi_max': chi, 'svd_min': 1.e-15, 'trunc_cut': 1e-8},
        'update_env': 10,
        'max_E_err': max_err, #precision in energy 
        'max_S_err': max_err, #precision in entropy
        'min_sweeps': 20,
        'max_sweeps': 1000,
        'mixer': True,
        "mixer_params": {"amplitude": 1.e-5, #Avoid local minima but do not influence results
                    "decay":1.2, #spreading function
                    "disable_after":30}
    }
    
    #Parameters of the model
    model_params = dict(L = 2, delta = delta, U = U, V = V, mu = mu, t = t, bc_MPS= "infinite", cons_Sz="None")
    
    #Generate model with initial parameters
    model = spin_chain_eff_model(model_params) 

    #Initial state ansatz (particle-conservation breaking state)
    psi = MPS.from_product_state(model.lat.mps_sites(), init_state * 2, model.lat.bc_MPS)

    engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    engine.run() #Runs iDMRG

    ################ Calculate observables and store results #################
    #For the correlations, first, define the reference point (idx0) and range (idx_list) 
    idx0 = 0 
    idx_list = np.arange(1, 100, 1) #in this case there is a single site per unit cell

    #Spin-spin correlations
    corrSz = np.zeros((len(idx_list)+1))
    corrSz[0] = psi.expectation_value_term([("Sz", idx0), ("Sz", idx0)])
    corrSz[1:] = psi.term_correlation_function_right([("Sz", 0)], [("Sz", 0)],\
                                                      i_L = idx0, j_R = idx_list).flatten()

    #Local density and density variance
    Sz = psi.expectation_value(["Sz"]).flatten()
    Sz_variance = np.array(psi.expectation_value(["Sz Sz"]).flatten()) - Sz**2

    results = {
        "model_params": model_params,
        "dmrg_params": dmrg_params, 
        "init_state": init_state,
        "corrSz": corrSz,
        "idx_list": idx_list,
        "Sz": Sz,
        "Sz_variance": Sz_variance,
        "entanglement_entropy": psi.entanglement_entropy(),
        "entanglement_spectrum": psi.entanglement_spectrum(),
        "corr_length": psi.correlation_length(),
        "iDMRG_convergence": engine.sweep_stats,
    }
    m_energy_MPO(results, psi, model, simulation = engine, results_key='energy')

    return results

if __name__ == "__main__":
    #Run the iDMRG algorithm for the effective spin model and store the results.

    #Parameters of the model
    delta = float(sys.argv[1]) #Value of delta = 2|mu| + U + 2V 
    t = float(sys.argv[2]) #nearest neighbour hopping amplitude
    mu = -1. #chemical potential
    r = float(sys.argv[3]) #ratio r=V/U
    U = (delta - 2)/(2*r +1)
    V = r*U

    #iDMRG parameters
    max_err = 1e-4 # precision in energy and entropy
    chi = int(sys.argv[4]) #max bond dimension: min(cutoff, Hilbert space size)

    #Initial state ansatz
    init_state = [np.array([1, 1])/np.sqrt(2.)]

    filename = 'effmodeliDMRG_delta_{:.3f}_t_{:.3f}_r_{:.3f}_chi_{:d}_maxerr_{:.0e}'.format(delta, t, r, chi, max_err)
    if not os.path.exists(filename + '.pkl'):
        results = run(chi, U, V, mu, t, delta, init_state, max_err)
        pickle.dump(results, open("./iDMRG_effmodel/" + filename + '.pkl', 'wb'))