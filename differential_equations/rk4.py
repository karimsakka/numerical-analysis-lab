import numpy as np

def runge_kutta_4(f, y0, t0, tn, n_steps):
    """
    Solves an ODE dy/dt = f(t, y) with initial condition y(t0) = y0
    using the 4th Order Runge-Kutta method.
    """
    t = np.linspace(t0, tn, n_steps + 1)
    h = (tn - t0) / n_steps
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * f(t[i] + h, y[i] + k3)
        
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        
    return t, y

if __name__ == "__main__":
    # Example ODE: dy/dt = -2 * y * t with y(0) = 1 (Exact solution: y = exp(-t^2))
    def sample_ode(t, y):
        return -2 * t * y

    t0, tn = 0.0, 2.0
    y0 = 1.0
    n_steps = 20

    t_vals, y_vals = runge_kutta_4(sample_ode, y0, t0, tn, n_steps)
    
    print("t values:", t_vals[:5])
    print("RK4 numerical solution:", y_vals[:5])
