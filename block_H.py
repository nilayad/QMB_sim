import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import sympy as smp
from collections import defaultdict

def apply_sx_i(s, i,N):
    """Flip the i-th spin in state s"""
    return s ^ (1 << i)
def apply_sz_i(s,i,N):
    return (s>>i)&1

def apply_s_up(s,i,N):
    loc_s =(s>>i)&1
    if loc_s == 0:
        return apply_sx_i(s,i,N)

    else :
        return None

def apply_s_down(s,i,N):
    loc_s =(s>>i)&1
    if loc_s == 1:
        return apply_sx_i(s,i,N)

    else :
        return None

def flip(s, i, N):
    """Flip the bits of the state `s` at positions i and (i+1)%N."""
    return s ^ (1 << i | 1 << ((i+1) % N)) # sigxi*sigxi+1

def count_ones(s, N):
    """Count the number of `1` in the binary representation of the state `s`."""
    return bin(s).count('1')    #sigz
    
def bin_N(s, N):
    """binary representation of integer s for N spins."""
    return bin(s)[2:].zfill(N)


def translate(s, N):
    """Shift the bits of the state `s` one position to the right (cyclically for N bits)."""
    return (s >> 1) | ((s & 1) << (N - 1))

def parity(s,N):
    if (-N+count_ones(s,N))%2==0:
        return 1
    else:
        return -1


def period(s,N):
    
    ts=translate(s,N)
    a=1
    while ts!=s:
        ref=ts
        ts=translate(ts,N)
        a+=1
        if a>N:
            return -1
    return a

def is_lex_representative(s, N):
    t = s
    for _ in range(N - 1):
        t = translate(t, N)
        if t < s:
            return False
    return True

def is_representative(a, k, N):
    
    """Check if |a> is the representative for the momentum state.

    Returns -1 if |a> is not a representative.
    If |a> is a representative, return the periodicity Ra,
    i.e. the smallest integer Ra > 0 such that T**Ra |a> = |a>."""
    if not is_lex_representative(a, N):
        
        return -1
        
    ra = period(a,N)
    if k%(N//ra)==0:
        
        #print('|a> is a representative state')
        
        return ra

    else:
        return -1



def get_representative(a, N):
    """Find the representative r in the orbit of s and return (r, l) such that |r>= T**l|a>"""
    r = a
    t = a
    l = 0
    for i in range(N):
        t = translate(t, N)
        if (t < r):
            r = t
            l = i + 1
    return r, l



def calc_basis(N):
    """Determine the (representatives of the) basis for each block.

    A block is detemined by the quantum numbers `qn`, here simply `k`.
    `basis` and `ind_in_basis` are dictionaries with `qn` as keys.
    For each block, `basis[qn]` contains all the representative spin configurations `sa`
    and periodicities `Ra` generating the state
    ``|a(k)> = 1/sqrt(Na) sum_l=0^{N-1} exp(i k l) T**l |sa>``

    `ind_in_basis[qn]` is a dictionary mapping from the representative spin configuration `sa`
    to the index within the list `basis[qn]`.
    """
    basis = dict()
    ind_in_basis = dict()
    for sa in range(2**N):
        for k in range(-N//2+1, N//2+1):
            qn = k
            Ra = is_representative(sa, k, N)
            if Ra > 0:
                if qn not in basis:
                    basis[qn] = []
                    ind_in_basis[qn] = dict()
                ind_in_basis[qn][sa] = len(basis[qn])
                basis[qn].append((sa, Ra))
    return basis, ind_in_basis

def calc_basis_kp(N):
    """Determine the (representatives of the) basis for each block.

    A block is detemined by the quantum numbers `qn`, here simply `k`.
    `basis` and `ind_in_basis` are dictionaries with `qn` as keys.
    For each block, `basis[qn]` contains all the representative spin configurations `sa`
    and periodicities `Ra` generating the state
    ``|a(k)> = 1/sqrt(Na) sum_l=0^{N-1} exp(i k l) T**l |sa>``

    `ind_in_basis[qn]` is a dictionary mapping from the representative spin configuration `sa`
    to the index within the list `basis[qn]`.
    """
    basis = dict()
    ind_in_basis = dict()
    for sa in range(2**N):
        
        for k in range(-N//2+1, N//2+1):
            
            Ra = is_representative(sa, k, N)
            p= parity(sa,N)
            qn = (k,p)
            if Ra > 0:
                if qn not in basis:
                    basis[qn] = []
                    ind_in_basis[qn] = dict()
                ind_in_basis[qn][sa] = len(basis[qn])
                basis[qn].append((sa, Ra))
    return basis, ind_in_basis
    
def basis_ksec(N,k_lis,symm_kp):
    basis = dict()
    ind_in_basis = dict()
    for sa in range(2**N):
        
        for k in k_lis:
            
            Ra = is_representative(sa, k, N)
            p= parity(sa,N)
            if symm_kp == 'k':
                qn = k
            if symm_kp=='kp':
                qn = (k,p)
            if Ra > 0:
                if qn not in basis:
                    basis[qn] = []
                    ind_in_basis[qn] = dict()
                ind_in_basis[qn][sa] = len(basis[qn])
                basis[qn].append((sa, Ra))
    return basis, ind_in_basis


def calc_H(N, J, g):
    """Determine the blocks of the Hamiltonian as scipy.sparse.csr_matrix."""
    print("Generating Hamiltonian ... ", end="", flush=True)
    basis, ind_in_basis = calc_basis(N)
    H = {}
    for qn in basis:
        M = len(basis[qn])
        H_block_data = []
        H_block_inds = []
        a = 0
        for sa, Ra in basis[qn]:
            H_block_data.append(-g * (-N + 2*count_ones(sa, N)))
            H_block_inds.append((a, a))
            for i in range(N):
                sb, l = get_representative(flip(sa, i, N), N)
                if sb in ind_in_basis[qn]:
                    b = ind_in_basis[qn][sb]
                    Rb = basis[qn][b][1]
                    k = qn*2*np.pi/N
                    H_block_data.append(-J*np.exp(-1j*k*l)*np.sqrt(Ra/Rb))
                    H_block_inds.append((b, a))
                # else: flipped state incompatible with the k value, |b(k)> is zero
            a += 1
        H_block_inds = np.array(H_block_inds)
        H_block_data = np.array(H_block_data)
        H_block = scipy.sparse.csr_matrix((H_block_data, (H_block_inds[:, 0], H_block_inds[:, 1])),
                                          shape=(M,M),dtype=complex)
        H[qn] = H_block
    print("done", flush=True)
    return H


def calc_H_kp(N, J, g):
    """Determine the blocks of the Hamiltonian as scipy.sparse.csr_matrix."""
    print("Generating Hamiltonian ... ", end="", flush=True)
    basis, ind_in_basis = calc_basis_kp(N)
    H = {}
    for qn in basis:
        M = len(basis[qn])
        H_block_data = []
        H_block_inds = []
        a = 0
        for sa, Ra in basis[qn]:
            H_block_data.append(-g * (-N + 2*count_ones(sa, N)))
            H_block_inds.append((a, a))
            for i in range(N):
                sb, l = get_representative(flip(sa, i, N), N)
                if sb in ind_in_basis[qn]:
                    b = ind_in_basis[qn][sb]
                    Rb = basis[qn][b][1]
                    k = qn[0]*2*np.pi/N
                    H_block_data.append(-J*np.exp(-1j*k*l)*np.sqrt(Ra/Rb))
                    H_block_inds.append((b, a))
                # else: flipped state incompatible with the k value, |b(k)> is zero
            a += 1
        H_block_inds = np.array(H_block_inds)
        H_block_data = np.array(H_block_data)
        H_block = scipy.sparse.csr_matrix((H_block_data, (H_block_inds[:, 0], H_block_inds[:, 1])),
                                          shape=(M,M),dtype=complex)
        H[qn] = H_block
    print("done", flush=True)
    return H


def block_H_kp(N, J, g,kp):
    """Determine the blocks of the Hamiltonian as scipy.sparse.csr_matrix."""
    print("Generating Hamiltonianfor k,p block ... ", end="", flush=True)
    basis, ind_in_basis = basis_ksec(N,K_list)
    H = {}
    for qn in basis:
        M = len(basis[qn])
        H_block_data = []
        H_block_inds = []
        a = 0
        for sa, Ra in basis[qn]:
            H_block_data.append(-g * (-N + 2*count_ones(sa, N)))
            H_block_inds.append((a, a))
            for i in range(N):
                sb, l = get_representative(flip(sa, i, N), N)
                if sb in ind_in_basis[qn]:
                    b = ind_in_basis[qn][sb]
                    Rb = basis[qn][b][1]
                    k = qn[0]*2*np.pi/N
                    H_block_data.append(-J*np.exp(-1j*k*l)*np.sqrt(Ra/Rb))
                    H_block_inds.append((b, a))
                # else: flipped state incompatible with the k value, |b(k)> is zero
            a += 1
        H_block_inds = np.array(H_block_inds)
        H_block_data = np.array(H_block_data)
        H_block = scipy.sparse.csr_matrix((H_block_data, (H_block_inds[:, 0], H_block_inds[:, 1])),
                                          shape=(M,M),dtype=complex)
        H[qn] = H_block
    print("done", flush=True)
    return H
###########################################################################################
# create block operator from local
#________________________________________________________________________________________

def block_operator(N, op, op_site,basis=None, ind_in_basis=None ):
    print("Generating operator k,p block ... ", end="", flush=True)
    if basis is None or ind_in_basis is None:
        basis, ind_in_basis = calc_basis(N)
    
    block_op = {}
    for qn in basis:
        if isinstance(qn,tuple):
            k = qn[0] * 2*np.pi/N 

        else :
            k= qn*2*np.pi/N

        M = len(basis[qn])
        Op_data = []
        Op_inds = []

        for a, (sa, Ra) in enumerate(basis[qn]):
            sj = op(sa, op_site, N)  # apply op at site `op_site`

            if sj!= None:
                sb, l = get_representative(sj, N)
                if sb in ind_in_basis[qn]:
                    b = ind_in_basis[qn][sb]
                    Rb = basis[qn][b][1]
                    phase = np.exp(-1j * k * l)
                   
                    val =  phase * np.sqrt(Ra / Rb)

                    Op_data.append(val)
                    Op_inds.append((b, a))
        if len(Op_data) == 0:
            Op_block = scipy.sparse.csr_matrix((M, M), dtype=complex)
        else:
            Op_inds = np.array(Op_inds)
            Op_data = np.array(Op_data)
            Op_block = scipy.sparse.csr_matrix((Op_data, (Op_inds[:, 0], Op_inds[:, 1])),
                                               shape=(M, M), dtype=complex)
  
        block_op[qn] = Op_block
    print("done", flush=True)
    return block_op



def block_operator_all_sites(N, op, basis=None, ind_in_basis=None):
    print("Generating operator k,p block ... ", end="", flush=True)
    if basis is None or ind_in_basis is None:
        basis, ind_in_basis = calc_basis(N)

    # Initialize empty dict for sum of blocks
    block_op_sum = {}

    for site in range(N):
        # Compute block operator for op acting on this single site
        block_op_site = block_operator(N, op, site, basis, ind_in_basis)

        # Sum the blocks site-by-site
        for qn, block_mat in block_op_site.items():
            if qn not in block_op_sum:
                block_op_sum[qn] = block_mat
            else:
                block_op_sum[qn] += block_mat
    print("done", flush=True)
    return block_op_sum


#############################################################################################
#___________________________________________________________________________________________
#supporting functions
def build_index_map(basis):
    """
    Constructs a mapping from full Hilbert space basis states to
    their corresponding indices in each symmetry-reduced block.

    Parameters:
    -----------
    basis : dict
        Dictionary of the form {qn: [(rep_state, degeneracy), ...]}
        where `rep_state` is the representative integer of the basis state.

    Returns:
    --------
    ind_in_basis : dict
        Dictionary of the form {qn: {rep_state: block_index}}, mapping
        each basis state to its index in the block Hamiltonian.
    """
    ind_in_basis = {}
    for qn in basis:
        ind_in_basis[qn] = {
            rep_state: i for i, (rep_state, _) in enumerate(basis[qn])
        }
    return ind_in_basis


#############################################################################################

def H_to_state(state_list, qn, basis, ind_in_basis, block_H):
    """
    Apply the Hamiltonian block to a state represented as an integer.
    
    Parameters:
        state_int: int - representative state label (s)
        qn: quantum number (e.g., momentum sector k)
        basis: dict[qnum] -> list of (s, R)
        ind_in_basis: dict[qnum] -> dict[s -> index]
        block_H: dict[qnum] -> sparse H block
    
    Returns:
        result: list of (s_b, amplitude)
    """
    result = defaultdict(complex)
    Hk = block_H[qn]
    for state, amp in state_list:
        
        ind = build_index_map(basis)
        a = ind[qn][state]
        #a = ind_in_basis[qn][state]
        #a = full_to_block_map[qn][state]

        row = (Hk.T).getrow(a)  # sparse row
    
        
        for b, val in zip(row.indices, row.data):
            s_b = basis[qn][b][0]  # get representative int of state |b⟩
            result[s_b]+=  amp*val
    
    return list(result.items())



def op_to_state(state_list, qn, basis, ind_in_basis, op):
    """
    Apply the Hamiltonian block to a state represented as an integer.
    
    Parameters:
        state_int: int - representative state label (s)
        qn: quantum number (e.g., momentum sector k)
        basis: dict[qnum] -> list of (s, R)
        ind_in_basis: dict[qnum] -> dict[s -> index]
        block_H: dict[qnum] -> sparse H block
    
    Returns:
        result: list of (s_b, amplitude)
    """
    result = defaultdict(complex)
    opk = op[qn]
    for state, amp in state_list:
        
        ind = build_index_map(basis)
        a = ind[qn][state]
        #a = full_to_block_map[qn][state]
        row = (opk.T).getrow(a)  # sparse row
    
        
        for b, val in zip(row.indices, row.data):
            s_b = basis[qn][b][0]  # get representative int of state |b⟩
            result[s_b]+=  amp*val
    
    return list(result.items())



#############################################################################################
def normalize_state(state_list):
    """
    Normalize a quantum state represented as a list of (int_state, amplitude).
    
    Parameters:
        state_list: list of (int, complex) -- (sᵢ, ampᵢ)
        
    Returns:
        normalized_list: list of (int, complex) with unit norm
    """
    norm = np.sqrt(sum(abs(amp)**2 for st, amp in state_list))
    
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    
    return [(s, amp / norm) for s, amp in state_list]
def norm_state(state):
    norm = np.sqrt(state_overlap(state,state))
           
    return [(s,amp/norm) for s,amp in state]

def state_overlap(state1, state2):
    """
    Compute ⟨state1|state2⟩ where states are lists of (s, amplitude).
    
    Parameters:
        state1, state2 : list of (int, complex)
            Quantum states in representative int form with amplitudes.
    
    Returns:
        complex
            Overlap ⟨state1|state2⟩
    """
    # Convert to dictionaries for efficient lookup
    dict1 = dict(state1)
    dict2 = dict(state2)
    
    overlap = 0.0 + 0.0j
    for s, amp2 in dict2.items():
        amp1 = dict1.get(s, 0.0)
        overlap += np.conj(amp1) * amp2
    
    return overlap

def state_lin_combination(s1,c1,s2,c2):
    result=defaultdict(complex)
    for sa,ampa in s1:
        result[sa] +=c1*ampa
    for sb,ampb in s2:
        result[sb]=c2*ampb

    return [(s,amp) for s,amp in result.items()]
