import numpy as np
import matplotlib.pyplot as plt

# Define the function
f = lambda x: np.exp(x) - 2*x - 3

# Bisection Method
def bisection_method(func, a, b, tolerance=1e-6, max_iterations=100):
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have opposite signs at endpoints a and b.")

    iterations = 0
    while (b - a) / 2 > tolerance:
        iterations += 1
        midpoint = (a + b) / 2
        if func(midpoint) == 0 or (b - a) / 2 < tolerance:
            return midpoint, iterations

        if func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

        if iterations >= max_iterations:
            raise ValueError("Bisection method did not converge within the maximum number of iterations.")

    return (a + b) / 2, iterations

# Secant Method
def secant_method(func, x0, x1, tolerance=1e-6, max_iterations=100):
    iterations = 0
    while abs(x1 - x0) > tolerance:
        iterations += 1
        f_x0 = func(x0)
        f_x1 = func(x1)

        if f_x1 - f_x0 == 0:
            raise ValueError("Division by zero in the secant method.")

        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        x0, x1 = x1, x2

        if iterations >= max_iterations:
            raise ValueError("Secant method did not converge within the maximum number of iterations.")

    return x1, iterations

# Define the interval and initial guesses
interval_a, interval_b = 0, 2
initial_guess1, initial_guess2 = 0, 2

# Exact root using Bisection Method
root_bisection, iterations_bisection = bisection_method(f, interval_a, interval_b)

# Exact root using Secant Method
root_secant, iterations_secant = secant_method(f, initial_guess1, initial_guess2)

# Calculate the exact root using a high-precision numerical solver
from scipy.optimize import fsolve
exact_root = fsolve(f, 1)[0]

# Calculate relative errors
relative_error_bisection = abs((exact_root - root_bisection) / exact_root)
relative_error_secant = abs((exact_root - root_secant) / exact_root)

# Print results
print("Bisection Method:")
print("  Root:", root_bisection)
print("  Iterations:", iterations_bisection)
print("  Relative Error:", relative_error_bisection)

print("Secant Method:")
print("  Root:", root_secant)
print("  Iterations:", iterations_secant)
print("  Relative Error:", relative_error_secant)

print("Exact Root (high precision):", exact_root)

# Comparison of methods
if iterations_bisection < iterations_secant:
    print("Bisection method is more efficient in terms of iterations.")
else:
    print("Secant method is more efficient in terms of iterations.")
