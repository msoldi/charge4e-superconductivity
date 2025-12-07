import numpy as np
import os, sys, pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tenpy.networks.site import GroupedSite, SpinHalfFermionSite
from tenpy.networks.mps import TransferMatrix
from tenpy.linalg.np_conserved import tensordot
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.mpo import MPO

"""
Phyton script to reproduce the transfer matrix data in the article "Charge-4e Superconductivity in a Hubbard model" 
(arXiv:2312.13348)
Authors:  Martina O. Soldini, Mark H. Fischer, and Titus Neupert
Affiliation: Physics Institute, University of Zurich, Winterthurerstrasse 190, 8057 Zurich, Switzerland
"""

plt.rc('font', size = 8)
plt.rc("pdf", fonttype=42)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

def get_grids(structure):
    if structure=="4e":
        grid2a = [[["Cdu Cdd"]], [["Id"]]]
        grid2b = [[["Cdu"]], [["Cdd"]]]
        grid4 = [[["Cdu Cdd"]], [["Cdu Cdd"]]]
        return [grid2a, grid2b, grid4]
    elif structure=="2e":
        grid2a = [[["Cdu Cdd"]], [["Id"]]]
        return [grid2a]

def T_overlaps(psi, grid_list):
    """
        This function computes the spectrum of the transfer matrix (TM) and the overlap betwen 
        the sub-leading eigenvector and some operators (given by the grid_list).
        Since we run a two-site iDMRG algorithm, we evaluate the overlaps first on the first rung,
        and then on the second rung.
        The transfer matrix can be decomposed in terms of its eigenvalues (lambda_i) and eigenvectors (vi)
        ---|B0 |---|B1 |---   ---|   |---                       ---|  |  |  |---
            |       |      =    | T |     = \sum_i (lambda_i)     |vi|  |vi|  
        ---|B0*|---|B1*|---   ---|   |---                       ---|  |  |  |---
        Where B0, B2 are the two B matrices returned by the iDMRG algorithm.
        The overlap of an operator A with the sub-leading eigenstate of the transfer matrix, v2, 
        is given by
            |  |---|B0 |---|B1 |---
            |  |     |       |     |
            |vi|    |A|      |     | = <v2|A>
            |  |     |       |     |     
            |  |---|B0*|---|B1*|---   
        Where on the right hand site we trace out the B1 matrices. Similarly, A can be contracted between
        the B1 tensors.
    """
    # Get the TM
    T = TransferMatrix(psi, psi, form="B", transpose = True) 
        #Note that from the two-site iDMRG algorithm we get the TM squared!
    #Transfer matrix leading eigenvector
    evals, evecs = T.eigenvectors(4, which="LM")
    if len(evals)<2:
        print("Bond dimension 1.")
        return
    evals = np.sqrt(np.abs(evals)) #Convert to eigenvalues of the TM instead of TM squared
    v = evecs[np.argsort(np.abs(evals))[-2]] # v = Sub-leading eigenvector (v2)

    #Compute expectation value with the two sets of B matrices (corresp. to the two-site iDMRG)
    overlaps_4e = []
    site = SpinHalfFermionSite(cons_N = "None", cons_Sz = "None")

    ### Overlap evaluated on the first rung
    B0, B1 = psi.get_B(0), psi.get_B(1) #Get the B matrices from psi.
    #Partially contract the B matrices.
    B0 = B0.replace_label('p', 'p1')
    B1 = B1.replace_label('p', 'p2')
    B = tensordot(B0, B1, axes=('vR', 'vL'))
    Bstar = tensordot(B0.iconj(), B1.iconj(), axes=('vR*', 'vL*'))  
    BB = tensordot(Bstar, B, axes =(["vR*"], ["vR"]))
    site = SpinHalfFermionSite(cons_N = "None", cons_Sz = "None")
    #Compute <v2|A> overlaps, with A given by the grid_list operators.
    overlaps = []
    for grids in grid_list:
        m = MPO.from_grids([site]*2, grids, bc="segment")
        W0, W1 = m._W[0], m._W[1].replace_labels(['p', 'p*'], ['q', 'q*'])
        W = tensordot(W0, W1, axes = (["wR"], ["wL"]))
        A = tensordot(v, BB.combine_legs(["vL*", "vL"], qconj=[+1]), axes = (['(vR*.vR)'], ['(vL*.vL)']))
        A = tensordot(W, A, axes = [["p*", "q*", "p", "q"], ["p1*", "p2*", "p1", "p2"]])
        overlaps.extend(np.abs(A.to_ndarray()).flatten().flatten())
    overlaps_4e.append(np.array(overlaps).flatten())

    ### Overlap evaluated on the second rung
    B0, B1, B2, B3 = psi.get_B(0), psi.get_B(1), psi.get_B(2), psi.get_B(3)
    B0 = B0.replace_label('p', 'p0')
    B1 = B1.replace_label('p', 'p1')
    B2 = B2.replace_label('p', 'p2')
    B3 = B3.replace_label('p', 'p3')

    B1, B2 = psi.get_B(0), psi.get_B(1)
    B1 = B1.replace_label('p', 'p1')
    B2 = B2.replace_label('p', 'p2')
    B = tensordot(B1, B2, axes=('vR', 'vL'))
    Bstar = tensordot(B1.iconj(), B2.iconj(), axes=('vR*', 'vL*'))  
    BB = tensordot(Bstar, B, axes =(["vR*"], ["vR"]))
    overlaps = []
    for grids in grid_list:
        m = MPO.from_grids([site]*2, grids, bc="segment")
        W0, W1 = m._W[0], m._W[1].replace_labels(['p', 'p*'], ['q', 'q*'])
        W = tensordot(W0, W1, axes = (["wR"], ["wL"]))
        A = tensordot(v, BB.combine_legs(["vL*", "vL"], qconj=[+1]), axes = (['(vR*.vR)'], ['(vL*.vL)']))
        A = tensordot(W, A, axes = [["p*", "q*", "p", "q"], ["p1*", "p2*", "p1", "p2"]])
        overlaps.extend(np.abs(A.to_ndarray()).flatten().flatten())
    overlaps_4e.append(np.array(overlaps).flatten())

    Tresults = {"evals": evals, "overlaps": overlaps_4e}
    return Tresults

if __name__ == "__main__":

    structure = sys.argv[1] #Input "4e" or "2e", for the 4e and 2e model respectively
    if structure == "4e":
        os.chdir("./iDMRG_charge4e")
    elif structure == "2e":
        os.chdir("./iDMRG_charge2e")
    else:
        print("structure not recognized")
        sys.exit()      

    #Set of parameters for which to compute the transfer matrix overlaps
    vals = [[-0.5, 0.6, 0.2], [0, 1., 0.2], [-0.4, 1.4, 0.2]] if structure=="4e" else [[-0.3, 0.5, 0]] #r!=0 for structure="2e"
    maxerr = 1e-5 #Consider data at fixed maxerr=max_E_err
    
    for delta, t, r in vals:
        #Collect data at different bond dimension chi, with the same delta, t, r=V/U
        param_string = "delta_{:.3f}_t_{:.3f}_r_{:.3f}".format(delta, t, r)
        files = [f for f in os.listdir(".") if f.endswith(".pkl") and f.__contains__(param_string)]
        if len(files)<1:
            continue
        resultsTM = {"t":t, "delta":delta, "r":r, "chis": {}}
        for f in files:
            with open(f, "rb") as file:
                resultiDMRG = pickle.load(file)
            
            chi = resultiDMRG["iDMRG_convergence"]["max_chi"][-1]
            
            #Compile data at same max_E_err
            max_err = resultiDMRG["dmrg_params"]["max_E_err"]
            if max_err != maxerr:
                continue
            #Check convergence of the iDMRG run
            if resultiDMRG["iDMRG_convergence"]["sweep"][-1] == resultiDMRG["dmrg_params"]["max_sweeps"]:
                print("Not converged")
                continue
            #Check that the state has bond dimension > 1
            if chi < 2:
                print("Bond dimension 1 (skip data point)")
                continue

            resultsTM["chis"][chi] = T_overlaps(psi=resultiDMRG['psi'], grid_list=get_grids(structure))

        fileTM = "../TransferMatrix/TMoverlaps{}_delta_{:.3f}_t_{:.3f}_r_{:.3f}.pkl".format(structure, delta, t, r)
        with open(fileTM, "wb") as file:
                pickle.dump(resultsTM,  file)