import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iterations=100):
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    
    for it in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
            
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {it + 1} iterations.")
            return x_new
        x = x_new
        
    raise ValueError("Solution did not converge within maximum iterations.")

if __name__ == "__main__":
    A = np.array([[4.0, 1.0, 2.0],
                  [3.0, 5.0, 1.0],
                  [1.0, 1.0, 3.0]])
    b = np.array([4.0, 7.0, 3.0])
    
    solution = gauss_seidel(A, b)
    print("Solution:", solution)
