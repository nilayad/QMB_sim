import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from numpy.linalg import svd
import psutil
import os
process = psutil.Process(os.getpid())
import time
import gc
import numba
from numba import jit


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from numpy.linalg import svd


# === Simulation Core ===

def initial_state(L):
    psi = np.zeros(2**L, dtype=complex)
    psi[0] = 1.0
    return psi

def apply_two_qubit_unitary(psi, U, i, j, L):
    psi_tensor = psi.reshape([2]*L)
    perm = [i, j] + [k for k in range(L) if k != i and k != j]
    psi_tensor = np.transpose(psi_tensor, perm)
    psi_matrix = psi_tensor.reshape((4, -1))
    psi_matrix = U @ psi_matrix
    psi_tensor = psi_matrix.reshape([2, 2] + [2]*(L-2))
    inverse_perm = np.argsort(perm)
    psi_tensor = np.transpose(psi_tensor, inverse_perm)
    return psi_tensor.reshape(2**L)

def projective_measure(psi, qubit, L):

    nz_indices = np.nonzero(psi)[0]
    if len(nz_indices) == 0:
        return psi  
        
    bit_vals = (nz_indices >>(L-qubit-1)) & 1
    
    mask0_nz = bit_vals == 0
    mask1_nz = bit_vals == 1
    prob0 = np.sum(np.abs(psi[nz_indices[mask0_nz]])**2)
    prob1 = np.sum(np.abs(psi[nz_indices[mask1_nz]])**2)
    
    if prob0 + prob1 == 0:
        return psi 
        
    outcome = np.random.choice([0, 1], p=[prob0, prob1])
    mask_nz = mask0_nz if outcome == 0 else mask1_nz
    psi_new = np.zeros_like(psi)
    indices_to_keep = nz_indices[mask_nz]
    psi_new[indices_to_keep] = psi[indices_to_keep]
    norm = np.linalg.norm(psi_new)
    
    return psi_new / norm if norm != 0 else psi

def entropy_half_chain(psi, L):
    psi = psi.reshape((2**(L//2), 2**(L//2)))
    s = svd(psi, compute_uv=False)
    probs = s**2
    probs = probs[probs > 1e-12]
    return -np.sum(probs * np.log(probs))

def simulate(L, depth, p, n_avg):
    entropies = []
    for avrun in range(n_avg):
             
        psi = initial_state(L)
        for t in range(depth):
            
            for i in range(0, L - 1, 2):
                U = unitary_group.rvs(4)
                psi = apply_two_qubit_unitary(psi, U, i, i+1, L)
            for i in range(L):
                if np.random.rand() < p:
                    psi = projective_measure(psi, i, L)
            for i in range(1, L - 1, 2):
                U = unitary_group.rvs(4)
                psi = apply_two_qubit_unitary(psi, U, i, i+1, L)
            
        S = entropy_half_chain(psi, L)
        
        entropies.append(S)
    return np.mean(entropies) / L




# === Plotting Code ===

if __name__ == "__main__":
    

    L = 14      # number of qubits
    depth = 20      # circuit depth
    n_avg = 20       # number of runs to average over
    
    p_vals = np.linspace(0.0, 1.0, 20)
    entropy_vals = []
    
    print("Running Simulation ... ", end="", flush=True)
    for p in p_vals:
        t0 = time.time()
        print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
        print(f"p = {p:.2f}", end="\r")
        S = simulate(L, depth, p, n_avg)
        
        entropy_vals.append(S)
        print("done, stored values, time taken = ",(time.time()-t0), flush=True)
        del S
        gc.collect()
    print("done", flush=True)
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(p_vals, entropy_vals, marker='o', label=f'L={L}')
    plt.xlabel('Measurement Rate p')
    plt.ylabel('⟨S(L/2)⟩ / L')
    plt.title('Measurement-Induced Phase Transition')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
