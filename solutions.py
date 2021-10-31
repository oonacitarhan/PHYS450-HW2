from qutip import *
from math import *

def q1():
    lambda1 = (1+1.0j)/sqrt(3)
    lambda2 = 1/sqrt(3)
    psi = lambda1*basis(4,0) + lambda2*basis(4,3)
    dm_psi = ket2dm(psi)
    print("q1:")
    print("Relative entropy of coherence: ", entropy_vn(dm_psi, base=2))
    print(dm_psi-qeye(4)*dm_psi)

def main():
    print("")
    q1()

main()
