import numpy as np

def jacobi(A, b, x0=None, tol=1e-6, max_iterations=100):
    """
    Solves the linear system Ax = b using the Jacobi iterative method.
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for it in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Jacobi converged in {it + 1} iterations.")
            return x_new
        x = x_new
        
    raise ValueError("Jacobi method did not converge within maximum iterations.")

if __name__ == "__main__":
    A = np.array([[10.0, -1.0, 2.0],
                  [-1.0, 11.0, -1.0],
                  [2.0, -1.0, 10.0]])
    b = np.array([6.0, 25.0, -11.0])
    
    sol = jacobi(A, b)
    print("Solution:", sol)
