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
import chebychev
process = psutil.Process(os.getpid())


def chebychev_recurence(H,v0,N):
    t0 = v0
    t1 = H@v0
    mu = [(v0.conj().T@t0)]
    mu.append(v0.conj().T@t1)
    tn_1 = t1
    tn_2=t0
    for n in range(2,N):
        tn = 2*H@tn_1-tn_2
        mu_n = v0.conj().T@tn
        mu.append(mu_n)
        tn_1,tn_2 = tn,tn_1

    return mu

def energy_bounds(H):
    emax = linal.eigsh(H,k=1,which='LA',return_eigenvectors=False)[0]
    emin = linal.eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]
    a = (emax-emin)/2
    b = (emax+emin)/2

    return a,b


def rescale_H(H):
    emax = linal.eigsh(H,k=1,which='LA',return_eigenvectors=False)[0]
    emin = linal.eigsh(H,k=1,which='SA',return_eigenvectors=False)[0]
    a = (emax-emin)/2
    b = (emax+emin)/2
    H_rescaled = (H-b*sp.eye(H.shape[0]))/a
    
    return H_rescaled



def compute_mu(H,O,psi0,N):
    H_til = rescale_H(H)
    v0 = O@psi0
    v0 = v0/(linal.norm(v0))
    
    return chebychev_recurence(H_til,v0,N)

def reconstruct_spectralfunc(mu,g,omega_list):
     
    mu = [i.toarray().flatten()[0] for i in mu] 
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



if __name__ == "__main__":
    
    L =14
    g=0.1
    j =1
    sx_list = lanczos.gen_sx_list(L)
    sz_list = lanczos.gen_sz_list(L)
    sy_list = lanczos.gen_sy_list(L)
    H1 = lanczos.gen_hamiltonian_heisenberg(sx_list,sz_list,sy_list,g,j,pbc =True)
    del sx_list,sy_list,sz_list
    gc.collect()
    #H2 = lanczos.gen_hamiltonian(sx_list,sz_list,g,j,pbc =True)
    O2 = operator_full(lanczos.Sx,L)
    #O1 = lanczos.singlesite_to_full(lanczos.Sz,5,L) +lanczos.singlesite_to_full(lanczos.Sz,6,L)
    psi01 = sp.lil_matrix((2**L,1), dtype=np.complex128)
    psi01[500,0] = 1.0
    # psi01 = np.zeros(2**L)
    # psi01[500]=1
    #psi0 = haar_stat(L,complex_entry=True)
    
    print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
    
    
    N = [40,100,300,450,600]
    for n in N:
        t0 = time.time()
        mu = chebychev.compute_mu(H1,O2,psi01,n)
    
        omega_list =  np.linspace(-1 + 1e-6, 1 - 1e-6, num=500)
        gn = chebychev.jackson_kernel(n)
        A_omega = chebychev.reconstruct_spectralfunc(mu,gn,omega_list)
    
        plt.plot(omega_list,A_omega,':', label =f'N={n}')
        print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
        del mu, gn, A_omega, omega_list
        gc.collect()
        print('time=', t0-time.time())
        print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")
    
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$A(\omega)$')
    plt.ylim(0,5)
    plt.legend()
    plt.title("Chebyshev Spectral Function")
    
    
    
    plt.show()

    print(f"Memory: {process.memory_info().rss / 1e6:.2f} MB")