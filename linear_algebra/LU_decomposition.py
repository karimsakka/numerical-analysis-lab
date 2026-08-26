import numpy as np

def lu_decomposition(A):
    """
    Computes LU Decomposition for a square matrix A such that A = L * U.
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, :] -= factor * U[i, :]
            
    return L, U

if __name__ == "__main__":
    A = np.array([[2.0, -1.0, -2.0],
                  [-4.0, 6.0, 3.0],
                  [-4.0, -2.0, 8.0]])
    
    L, U = lu_decomposition(A)
    print("L matrix:\n", L)
    print("U matrix:\n", U)
