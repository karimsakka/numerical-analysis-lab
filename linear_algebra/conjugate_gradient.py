import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iterations=100):
    """
    Solves Ax = b for symmetric positive-definite matrix A using Conjugate Gradient.
    """
    x = np.zeros_like(b, dtype=float) if x0 is None else x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = np.dot(r, r)
    
    for i in range(max_iterations):
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        
        if np.sqrt(rs_new) < tol:
            print(f"CG converged in {i + 1} iterations.")
            return x
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        
    return x

if __name__ == "__main__":
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    
    sol = conjugate_gradient(A, b)
    print("Solution:", sol)
