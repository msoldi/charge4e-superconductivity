import numpy as np
import os
import sys 
import pickle

from charge4e_iDMRG import  run

"""
Python script to reproduce the iDMRG data in the article "Charge-4e Superconductivity in a Hubbard model" (arXiv:2312.13348)
by Martina O. Soldini, Mark H. Fischer, and Titus Neuper
Affiliation: Physics Institute, University of Zurich, Winterthurerstrasse 190, 8057 Zurich, Switzerland

This script launches the iDMRG calculations for the charge-2e model.
"""

if __name__ == "__main__":
    #Run the iDMRG algorithm for the 2e model.
    #Parameters of the 2e model
    t = float(sys.argv[1]) #nearest neighbour hopping amplitude
    mu = -1. #chemical potential, fixed.
    U =  float(sys.argv[2])

    #iDMRG parameters
    L = 2 # two-site iDMRG
    chi = int(sys.argv[3]) #max bond dimension: min(cutoff, Hilbert space size)
    max_err = 1e-5
    #Initial state
    init_state = [np.array([1, 1, 1, 1])/2.] #Single orbital per unit cell

    model_params = {"t":t, "U":U, "L":L, "bc_MPS": "infinite"}

    filename = 'charge2eiDMRG_U_{:.3f}_t_{:.3f}_chi_{:d}_maxerr_{:.0e}'.format(U, t, chi, max_err)
    if not os.path.exists(filename + '.pkl'):
        results = run(which_model="2e", model_params=model_params, chi=chi, init_state=init_state, max_err=max_err)
        with open("./iDMRG_charge2e/" + filename + '.pkl', 'wb') as f:
            pickle.dump(results, f)

