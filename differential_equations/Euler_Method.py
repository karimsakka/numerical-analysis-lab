import numpy as np

def euler_method(f, y0, t0, tn, n_steps):
    """
    Solves an ODE dy/dt = f(t, y) with initial condition y(t0) = y0
    using the Explicit Euler method.
    """
    t = np.linspace(t0, tn, n_steps + 1)
    h = (tn - t0) / n_steps
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        
    return t, y

if __name__ == "__main__":
    # Example ODE: dy/dt = -2 * y * t with y(0) = 1 (Exact solution: y = exp(-t^2))
    def sample_ode(t, y):
        return -2 * t * y

    t0, tn = 0.0, 2.0
    y0 = 1.0
    n_steps = 20

    t_vals, y_vals = euler_method(sample_ode, y0, t0, tn, n_steps)
    
    print("t values:", t_vals[:5])
    print("Euler numerical solution:", y_vals[:5])
