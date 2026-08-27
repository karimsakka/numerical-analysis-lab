import numpy as np
import matplotlib.pyplot as plt


def adams_bashforth_2step(f, x0, y0, x_end, h):
    """
    Solves an Initial Value Problem (IVP) using the 2-step Adams-Bashforth Explicit Method.
    Requires RK2/Euler for the first step initialization.
    """
    x_values = np.arange(x0, x_end + h, h)
    n = len(x_values)
    y_values = np.zeros(n)
    
    # Initial condition
    y_values[0] = y0
    
    # Step 1: Use RK2 (Heun's method) to find y1
    k1 = f(x_values[0], y_values[0])
    k2 = f(x_values[0] + h, y_values[0] + h * k1)
    y_values[1] = y_values[0] + 0.5 * h * (k1 + k2)
    
    # Multi-step loop for remaining points
    for i in range(1, n - 1):
        f_current = f(x_values[i], y_values[i])
        f_prev = f(x_values[i-1], y_values[i-1])
        
        # Adams-Bashforth 2-Step Formula
        y_values[i + 1] = y_values[i] + 0.5 * h * (3 * f_current - f_prev)
        
    return x_values, y_values


# Example Usage: Solving dy/dx = x - y
if __name__ == "__main__":
    def f(x, y):
        return x - y

    x0, y0 = 0.0, 1.0
    x_end = 5.0
    h = 0.1

    x_res, y_res = adams_bashforth_2step(f, x0, y0, x_end, h)

    print("Adams-Bashforth Numerical Results (First 5 steps):")
    for x, y in zip(x_res[:5], y_res[:5]):
        print(f"x = {x:.2f}, y = {y:.4f}")
