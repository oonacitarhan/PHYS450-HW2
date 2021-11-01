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

def main():
    print("")
    q1()
    print("")
    q2()

main()
