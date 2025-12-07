import numpy as np
import os
import sys 
import pickle

#Import Charge 4e model and iDMRG running function from "charge4e_iDMRG"
from charge4e_iDMRG import run
"""
Phyton script to reproduce susceptibility data in the article "Charge-4e Superconductivity in a Hubbard model" 
(arXiv:2312.13348)
Authors:  Martina O. Soldini, Mark H. Fischer, and Titus Neupert
Affiliation: Physics Institute, University of Zurich, Winterthurerstrasse 190, 8057 Zurich, Switzerland
"""
if __name__ == "__main__":

    #Parameters of the model
    delta = float(sys.argv[1]) #Value of delta = 2|mu| + U + 2V 
    t = float(sys.argv[2]) #nearest neighbour hopping amplitude
    mu = -1. #chemical potential
    r = float(sys.argv[3]) #ratio r=V/U
    U = (delta - 2)/(2*r +1)
    V = r*U

    #iDMRG parameters
    L = 2 # two-site iDMRG
    max_err =1e-5
    chi = int(sys.argv[4]) #max bond dimension: min(cutoff, Hilbert space size)

    #Initial state
    init_state = [np.array([1, 1, 1, 1])/2.] * 2 #Two orbitals per unit cell

    DeltaVals = np.linspace(0, 0.001, 11)
    for Del in DeltaVals:
        for Del2, Del4 in zip([Del, 0], [0, Del]):
            model_params = {"mu":mu, "t":t, "delta":delta, "U":U, "V":V, "L":L, "bc_MPS": "infinite", "Del2":Del2, "Del4":Del4}

            filename = 'charge4eiDMRGsusc_delta_{:.3f}_t_{:.3f}_r_{:.3f}_chi_{:d}_Del2_{:.5f}_Del4_{:.5f}_maxerr_{:.0e}'.format(delta, t, r, chi, Del2, Del4, max_err)
            if not os.path.exists(filename + '.pkl'):
                results = run(which_model="4e", model_params=model_params, chi=chi, init_state=init_state, max_err=max_err)
                
                #Compute expectation values of the order parameters as defined in arXiv:2312.13348
                # P = c^dag_{1, up}c^dag_{1, down} and Q=c^dag_{1, up}c^dag_{1, down}c^dag_{2, up}c^dag_{2, down}
                psi = results["psi"]
                results["Q"] = psi.expectation_value_term([("Cdd", 0), ("Cdu", 0), ("Cdd", 1), ("Cdu", 1)])
                results["P"] = [psi.expectation_value_term([("Cdd", 0), ("Cdu", j)]) for j in [0, 1]]

                with open("./iDMRG_charge4e_susceptibility/" + filename + '.pkl', 'wb') as f:
                    pickle.dump(results, f)