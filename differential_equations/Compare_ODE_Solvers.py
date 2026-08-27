import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x):
    """Exact analytical solution for dy/dx = x - y with y(0) = 1."""
    return x - 1 + 2 * np.exp(-x)


def euler_method(f, x0, y0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0
    for i in range(len(x_values) - 1):
        y_values[i + 1] = y_values[i] + h * f(x_values[i], y_values[i])
    return x_values, y_values


def runge_kutta_4(f, x0, y0, x_end, h):
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


def adams_bashforth_2step(f, x0, y0, x_end, h):
    x_values = np.arange(x0, x_end + h, h)
    n = len(x_values)
    y_values = np.zeros(n)
    y_values[0] = y0
    k1 = f(x_values[0], y_values[0])
    k2 = f(x_values[0] + h, y_values[0] + h * k1)
    y_values[1] = y_values[0] + 0.5 * h * (k1 + k2)
    for i in range(1, n - 1):
        f_current = f(x_values[i], y_values[i])
        f_prev = f(x_values[i - 1], y_values[i - 1])
        y_values[i + 1] = y_values[i] + 0.5 * h * (3 * f_current - f_prev)
    return x_values, y_values


# Compare solutions and plot error curves
if __name__ == "__main__":
    def f(x, y):
        return x - y

    x0, y0 = 0.0, 1.0
    x_end = 5.0
    h = 0.1

    x_e, y_e = euler_method(f, x0, y0, x_end, h)
    x_rk, y_rk = runge_kutta_4(f, x0, y0, x_end, h)
    x_ab, y_ab = adams_bashforth_2step(f, x0, y0, x_end, h)
    y_exact = exact_solution(x_e)

    # Plot Numerical Solutions vs Exact Solution
    plt.figure(figsize=(10, 5))
    plt.plot(x_e, y_exact, "k-", label="Exact Solution", linewidth=2)
    plt.plot(x_e, y_e, "r--", label="Euler Method")
    plt.plot(x_rk, y_rk, "b-.", label="RK4 Method")
    plt.plot(x_ab, y_ab, "g:", label="Adams-Bashforth 2-Step")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of ODE Solvers")
    plt.legend()
    plt.grid(True)
    plt.savefig("ode_methods_comparison.png")
    plt.show()
