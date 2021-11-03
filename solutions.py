"""
PHYS 450 Homework 2
by Özgün Ozan Nacitarhan

Abbreviations:
dm - Density matrix
arr - array
pt - partial trace
diag - diagonal
"""

from qutip import *
from math import *
import numpy as np
import scipy
from scipy.sparse.dia import dia_matrix

def q1():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    dm_psi_diag_arr = dm_psi_arr.copy()
    l1_norm = 0
    for i in range(len(dm_psi_arr)):
        for j in range(len(dm_psi_arr[0])):
            if not i == j:
                dm_psi_diag_arr[i][j] = 0
                l1_norm += np.abs(dm_psi_arr[i][j])
    dm_psi_diag = Qobj(dm_psi_diag_arr)
    print("q1:")
    print("Relative entropy of coherence: ", entropy_vn(dm_psi_diag, base=2) - entropy_vn(dm_psi, base=2))
    print("l1-norm of coherence: ", l1_norm)

def q2():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    pt_dm = (dm_psi_arr[0][0]+dm_psi_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_arr[2][2]+dm_psi_arr[3][3]) * ket2dm(basis(2, 1))
    print("q2")
    print("Tot. Corr. shared in psi: ",2*entropy_vn(pt_dm, base=2) - entropy_vn(dm_psi, base=2))

    # Diagonalization, a different method from q1
    dm_psi_diag = dm_psi_arr[3][3] * ket2dm(basis(4, 3))
    for i in range(3):
        dm_psi_diag += dm_psi_arr[i][i] * ket2dm(basis(4, i))   
    dm_psi_diag_arr = dm_psi_diag.full(order='C')
    pt_dm_diag = (dm_psi_diag_arr[0][0]+dm_psi_diag_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_diag_arr[2][2]+dm_psi_diag_arr[3][3]) * ket2dm(basis(2, 1))
    print("Tot. Corr. shared in qd: ",2*entropy_vn(pt_dm_diag, base=2) - entropy_vn(dm_psi_diag, base=2))

def q3():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    print("q3:")
    pauliY = Qobj(tensor([sigmay(),sigmay()]).data.toarray())
    # p' = p * tensor(pauliY,pauliY) * dagger(p) * tensor(pauliY,pauliY) s.t. p = denstiy matrix
    dm = dm_psi * pauliY * dm_psi.conj() * pauliY
    eigenvalues = (np.around(np.linalg.eigvals(dm.full(order='C')), decimals=5))
    eigenvalues_sorted = np.real(sorted(eigenvalues, reverse=True))
    c = sqrt(eigenvalues_sorted[0]) - sqrt(eigenvalues_sorted[1]) - sqrt(eigenvalues_sorted[2]) - sqrt(eigenvalues_sorted[3])
    f = (1+sqrt(1-c**2))/2
    Ef = -f*log(f,2)-(1-f)*log((1-f),2)
    print("Entanglement of Formation: ", Ef)

def q4():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    # could not code partial transpose
    # it is manually calculated and implemented
    partial_transpose_dm = Qobj([[2/3,0,0,0],[0,0,1/3+1/3j,0],[0,1/3-1/3j,0,0],[0,0,0,1/3]])
    eigenvalues = np.around(np.linalg.eigvals(partial_transpose_dm.full(order='C')), decimals=5)
    negativity = 0
    for e in eigenvalues:
        negativity += (abs(e)-e)/2
    print("q4:")
    print("Entanglement Negativity: ", np.real(negativity)) # sqrt(2)/3

def q5():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = 3/5*ket2dm(psi)
    dm_psi += 1/5*ket2dm(basis(4,2))
    dm_psi += 1/5*ket2dm(basis(4,1))
    dm_psi_arr = dm_psi.full(order='C')
    dm_psi_diag_arr = dm_psi_arr.copy()
    l1_norm = 0
    for i in range(len(dm_psi_arr)):
        for j in range(len(dm_psi_arr[0])):
            if not i == j:
                dm_psi_diag_arr[i][j] = 0
                l1_norm += np.abs(dm_psi_arr[i][j])
    dm_psi_diag = Qobj(dm_psi_diag_arr)
    print("q5:")
    print("Relative entropy of coherence: ", entropy_vn(dm_psi_diag, base=2) - entropy_vn(dm_psi, base=2))
    print("l1-norm of coherence: ", l1_norm)

def q6():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = 3/5*ket2dm(psi)
    dm_psi += 1/5*ket2dm(basis(4,2))
    dm_psi += 1/5*ket2dm(basis(4,1))
    dm_psi_arr = dm_psi.full(order='C')
    pt_dm = (dm_psi_arr[0][0]+dm_psi_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_arr[2][2]+dm_psi_arr[3][3]) * ket2dm(basis(2, 1))
    print("q6")
    print("Tot. Corr. shared in psi: ",2*entropy_vn(pt_dm, base=2) - entropy_vn(dm_psi, base=2))

    # Diagonalization, a different method from q1
    dm_psi_diag = dm_psi_arr[3][3] * ket2dm(basis(4, 3))
    for i in range(3):
        dm_psi_diag += dm_psi_arr[i][i] * ket2dm(basis(4, i))   
    dm_psi_diag_arr = dm_psi_diag.full(order='C')
    pt_dm_diag = (dm_psi_diag_arr[0][0]+dm_psi_diag_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_diag_arr[2][2]+dm_psi_diag_arr[3][3]) * ket2dm(basis(2, 1))
    print("Tot. Corr. shared in qd: ",2*entropy_vn(pt_dm_diag, base=2) - entropy_vn(dm_psi_diag, base=2))

def q7():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = 3/5*ket2dm(psi)
    dm_psi += 1/5*ket2dm(basis(4,2))
    dm_psi += 1/5*ket2dm(basis(4,1))
    dm_psi_arr = dm_psi.full(order='C')
    print("q7:")
    pauliY = Qobj(tensor([sigmay(),sigmay()]).data.toarray())
    # p' = p * tensor(pauliY,pauliY) * dagger(p) * tensor(pauliY,pauliY) s.t. p = denstiy matrix
    dm = dm_psi * pauliY * dm_psi.conj() * pauliY
    eigenvalues = (np.around(np.linalg.eigvals(dm.full(order='C')), decimals=5))
    eigenvalues_sorted = np.real(sorted(eigenvalues, reverse=True))
    c = sqrt(eigenvalues_sorted[0]) - sqrt(eigenvalues_sorted[1]) - sqrt(eigenvalues_sorted[2]) - sqrt(eigenvalues_sorted[3])
    f = (1+sqrt(1-c**2))/2
    Ef = -f*log(f,2)-(1-f)*log((1-f),2)
    print("Entanglement of Formation: ", Ef)

def q8():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = 3/5*ket2dm(psi)
    dm_psi += 1/5*ket2dm(basis(4,2))
    dm_psi += 1/5*ket2dm(basis(4,1))
    # could not code partial transpose
    # it is manually calculated and implemented
    partial_transpose_dm = Qobj([[2/5,0,0,0],[0,1/5,1/5+1/5j,0],[0,1/5-1/5j,1/5,0],[0,0,0,1/5]])
    eigenvalues = np.around(np.linalg.eigvals(partial_transpose_dm.full(order='C')), decimals=5)
    negativity = 0
    for e in eigenvalues:
        negativity += (abs(e)-e)/2
    print("q8:")
    print("Entanglement Negativity: ", np.real(negativity)) # sqrt(2)/3

def main():
    print("")
    q1()
    print("")
    q2()
    print("")
    q3()
    print("")
    q4()
    print("")
    q5()
    print("")
    q6()
    print("")
    q7()
    print("")
    q8()

main()
