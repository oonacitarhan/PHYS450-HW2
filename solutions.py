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

def q1():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    print("q1:")
    print("Relative entropy of coherence: ", entropy_vn(dm_psi, base=2))
    dm_psi_arr = dm_psi.full(order='C')
    l1_norm = 0
    for i in range(len(dm_psi_arr)):
        for j in range(len(dm_psi_arr[0])):
            if not i == j:
                l1_norm += np.abs(dm_psi_arr[i][j])
    print("l1-norm of coherence: ", l1_norm)

def q2():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    pt_dm = (dm_psi_arr[0][0]+dm_psi_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_arr[2][2]+dm_psi_arr[3][3]) * ket2dm(basis(2, 1))
    print("q2")
    print("Tot. Corr. shared in psi: ",2*entropy_vn(pt_dm) - entropy_vn(dm_psi))
    dm_psi_diag = dm_psi_arr[3][3] * ket2dm(basis(4, 3))
    for i in range(3):
        dm_psi_diag += dm_psi_arr[i][i] * ket2dm(basis(4, i))
    dm_psi_diag_arr = dm_psi_diag.full(order='C')
    pt_dm_diag = (dm_psi_diag_arr[0][0]+dm_psi_diag_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_diag_arr[2][2]+dm_psi_diag_arr[3][3]) * ket2dm(basis(2, 1))
    print("Tot. Corr. shared in qd: ",2*entropy_vn(pt_dm_diag) - entropy_vn(dm_psi_diag))

def q3():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    pt_dm = (dm_psi_arr[0][0]+dm_psi_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_arr[2][2]+dm_psi_arr[3][3]) * ket2dm(basis(2, 1))
    print("q3:")
    print("Entanglement Entropy: ", entropy_vn(pt_dm, base=2))

def q4():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    print("q4:")
    print("Entanglement Negativity: ", negativity(dm_psi, [basis(2,0),basis(2,1)], method='eigenvalues'))

def q5():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = sqrt(3/5)*lambda1*basis(4,0) + sqrt(1/5)*basis(4,1) + sqrt(1/5)*basis(4,2) + sqrt(3/5)*lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    print("q5:")
    print("Relative entropy of coherence: ", entropy_vn(dm_psi, base=2))
    dm_psi_arr = dm_psi.full(order='C')
    l1_norm = 0
    for i in range(len(dm_psi_arr)):
        for j in range(len(dm_psi_arr[0])):
            if not i == j:
                l1_norm += np.abs(dm_psi_arr[i][j])
    print("l1-norm of coherence: ", l1_norm)

def q6():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = sqrt(3/5)*lambda1*basis(4,0) + sqrt(1/5)*basis(4,1) + sqrt(1/5)*basis(4,2) + sqrt(3/5)*lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    pt_dm = (dm_psi_arr[0][0]+dm_psi_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_arr[2][2]+dm_psi_arr[3][3]) * ket2dm(basis(2, 1))
    print("q6")
    print("Tot. Corr. shared in psi: ",2*entropy_vn(pt_dm) - entropy_vn(dm_psi))
    dm_psi_diag = dm_psi_arr[3][3] * ket2dm(basis(4, 3))
    for i in range(3):
        dm_psi_diag += dm_psi_arr[i][i] * ket2dm(basis(4, i))
    dm_psi_diag_arr = dm_psi_diag.full(order='C')
    pt_dm_diag = (dm_psi_diag_arr[0][0]+dm_psi_diag_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_diag_arr[2][2]+dm_psi_diag_arr[3][3]) * ket2dm(basis(2, 1))
    print("Tot. Corr. shared in qd: ",2*entropy_vn(pt_dm_diag) - entropy_vn(dm_psi_diag))

def q7():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = sqrt(3/5)*lambda1*basis(4,0) + sqrt(1/5)*basis(4,1) + sqrt(1/5)*basis(4,2) + sqrt(3/5)*lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    dm_psi_arr = dm_psi.full(order='C')
    pt_dm = (dm_psi_arr[0][0]+dm_psi_arr[1][1]) * ket2dm(basis(2, 0)) + (dm_psi_arr[2][2]+dm_psi_arr[3][3]) * ket2dm(basis(2, 1))
    print("q7:")
    print("Entanglement Entropy: ", entropy_vn(pt_dm, base=2))

def q8():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = sqrt(3/5)*lambda1*basis(4,0) + sqrt(1/5)*basis(4,1) + sqrt(1/5)*basis(4,2) + sqrt(3/5)*lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    print("q8:")
    print("Entanglement Negativity: ", negativity(dm_psi, [basis(2,0),basis(2,1)], method='eigenvalues'))

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
