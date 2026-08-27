import numpy as np
import matplotlib.pyplot as plt


def runge_kutta_4(f, x0, y0, x_end, h):
    """
    Solves an Initial Value Problem (IVP) for ODEs using the Classical 4th-Order Runge-Kutta Method.
    
    Parameters:
    -----------
    f : function
            The derivative function dy/dx = f(x, y).
    x0 : float
            Initial value of x.
    y0 : float
            Initial value of y at x0.
    x_end : float
            Final value of x.
    h : float
            Step size.

    Returns:
    --------
    x_values : numpy array
            Grid points for x.
    y_values : numpy array
            Numerical solution for y at each x point.
    """
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    for i in range(len(x_values) - 1):
        x = x_values[i]
        y = y_values[i]

        k1 = f(x, y)
        k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(x + h, y + h * k3)

        y_values[i + 1] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_values, y_values


# Example Usage: Solving dy/dx = x - y with initial condition y(0) = 1
if __name__ == "__main__":
    def f(x, y):
        return x - y

    x0, y0 = 0.0, 1.0
    x_end = 5.0
    h = 0.1

    x_res, y_res = runge_kutta_4(f, x0, y0, x_end, h)

    print("Numerical Results (First 5 steps):")
    for x, y in zip(x_res[:5], y_res[:5]):
        print(f"x = {x:.2f}, y = {y:.4f}")
