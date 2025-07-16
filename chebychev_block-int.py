import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as linal
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import psutil
import os
import lanczos
import block_H



process = psutil.Process(os.getpid())

def chebychev_recurence(H,qn,v0,basis,ind_in_basis,N):
    t0 = v0
    t1 = block_H.H_to_state(v0,qn,basis,ind_in_basis,H)
    mu = [block_H.state_overlap(v0,t0)]
    mu.append(block_H.state_overlap(v0,t1))
    tn_1 = t1
    tn_2=t0
    for n in range(2,N):
        ht1 = block_H.H_to_state(tn_1,qn,basis,ind_in_basis,H)
        tn = block_H.state_lin_combination(ht1,2,tn_2,-1)
        mu_n = block_H.state_overlap(v0,tn)
        mu.append(mu_n)
        tn_1,tn_2 = tn,tn_1

    return mu

def energy_bounds(H):
    emax = linal.eigsh(H,k=1,which='LA',return_eigenvectors=False)[0]
    emin = linal.eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]
    a = (emax-emin)/2
    b = (emax+emin)/2

    return a,b
    
def rescale_H(H_block):
    a,b = energy_bounds(H_block)
    H_tilde = (H_block - b * sp.eye(H_block.shape[0], format='csr')) / a
    return H_tilde

# def rescale_H(H):
#     emax = linal.eigsh(H,k=1,which='LA',return_eigenvectors=False)[0]
#     emin = linal.eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]
#     a = (emax-emin)/2
#     b = (emax+emin)/2
#     H_rescaled = (H-b*sp.eye(H.shape[0]))/a
    
#     return H_rescaled



def compute_mu(H,O,psi0,qn,basis,ind_in_basis,N):
    h_block = H[qn]
    H_til = rescale_H(h_block)
    v0 = block_H.op_to_state(psi0,qn,basis,ind_in_basis,O)
    v0 = block_H.normalize_state(v0)
    
    return chebychev_recurence(H_til,qn,v0,basis,ind_in_basis,N)

def reconstruct_spectralfunc(mu,g,omega_list):
     
    #mu = [i.toarray().flatten()[0] for i in mu] 
    N = len(mu)
    A = np.zeros_like(omega_list)

    for i, omega in enumerate(omega_list):
        tn_omega = np.cos(np.arange(N) * np.arccos(omega))
        A[i] = np.dot(g * mu, tn_omega)  

    return A / (np.pi * np.sqrt(1 - omega_list**2 + 1e-10))

    # N = len(mu)
    # # if omega_list==None:
    # #     omega_list = np.linspace(-1 + 1e-6, 1 - 1e-6, num=3)
    # A = np.zeros_like(omega_list)
    # for i,omega in enumerate(omega_list):
    #     tn_omega = np.cos(np.arange(N)*np.arccos(omega))
    #     A[i] = (g*mu @ tn_omega)
    # return A/(np.pi * np.sqrt(1 - omega_list**2 + 1e-10))


def jackson_kernel(N):
    n = np.arange(N)
    return ((N - n + 1) * np.cos(np.pi * n / (N + 1)) +
            np.sin(np.pi * n / (N + 1)) / np.tan(np.pi / (N + 1))) / (N + 1)





##_________________________________________________________________________________________##

def operator_full(op,L):
    op_list = [op]*L # = [Id, Id, Id ...] with L entries
    
    full = op_list[0]
   
    for op_i in op_list[1:]:
        full = sp.kron(full, op_i, format="csr")

    return full


def haar_stat(L,complex_entry):
    dim = 2**L
    rng = np.random.default_rng()
    if complex_entry:
        rand_vec = rng.normal(0, 1, dim) + 1j * rng.normal(0, 1, dim)
    else:
        rand_vec = rng.normal(0, 1, dim)  # real-only Haar state
    rand_vec= rand_vec / np.linalg.norm(rand_vec)
   
    return sp.csc_matrix(rand_vec.reshape(-1,1),dtype=np.complex128)




#############################################################################################
#___________________________________________________________________________________________
#supporting functions
#############################################################################################







#############################################################################################


if __name__ == "__main__":
        
    L =18
    g=0.1
    j =1
    
    basis,indeces = block_H.calc_basis(L)
    k_sec= [i for i in basis.keys()]
    op_i = block_H.apply_sx_i
    print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
    H1 = block_H.calc_H(L,j,g)
    print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
    O1 = block_H.block_operator(L,op_i,3,basis,indeces)
    O1_full = block_H.block_operator_all_sites(L,op_i,basis,indeces)
    
    print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
        
    k_sector = [i for i in basis] 
    len_k = len(k_sector)
    k_list =[k_sector[0], k_sector[l//4],k_sector[l//3],k_sector[l//2],k_sector[-1]]
    i=0
    for k in k_list:
        i+=1
        plt.subplot(5,1,i)
        psi01 = [((basis[k][-5][0]),1)]
        
        N = [40,100,300,600]
        for i in N:
            t0 = time.time()
            mu = compute_mu(H1,O1,psi01,k,basis,indeces,i)
        
            omega_list =  np.linspace(-1 + 1e-6, 1 - 1e-6, num=500)
            gn = jackson_kernel(i)
            A_omega = reconstruct_spectralfunc(mu,gn,omega_list)
        
            plt.plot(omega_list,A_omega,':', label =f'N={i}')
        
            del mu, gn, A_omega, omega_list
            gc.collect()
            print('time=', t0-time.time())
            print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
        
        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$A(\omega)$')
        #plt.ylim(0,5)
        plt.legend()
        plt.title("Chebyshev Spectral Function")



    plt.show()

    print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
    
    
    