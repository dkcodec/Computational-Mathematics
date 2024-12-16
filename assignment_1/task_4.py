import numpy as np
import matplotlib.pyplot as plt
import cmath

# Define the function
f = lambda x: x**3 + x**2 + x + 1

# Muller's Method
def mullers_method(f, x0, x1, x2, tol=1e-6, max_iter=100):
    # Iterative root approximation by Muller's method
    for _ in range(max_iter):
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (f(x1) - f(x0)) / h0
        delta1 = (f(x2) - f(x1)) / h1
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f(x2)

        discriminant = cmath.sqrt(b**2 - 4*a*c)
        if abs(b + discriminant) > abs(b - discriminant):
            denominator = b + discriminant
        else:
            denominator = b - discriminant

        x3 = x2 - (2 * c) / denominator  # Formula

        if abs(x3 - x2) < tol:  # Stop condition
            return x3
        x0, x1, x2 = x1, x2, x3

    print("The method did not converge within the specified number of iterations.")
    return None

# Initial guesses
x0, x1, x2 = -1, 0, 1

# Find root and collect iteration details
try:
    root, iteration_details = mullers_method(f, x0, x1, x2)

    # Verify the result
    function_value = f(root)
    absolute_error = abs(function_value)

    # Create a table of iterations
    print("Iteration\tRoot Approximation\tAbsolute Error")
    for iteration in iteration_details:
        print(f"{iteration[0]}\t{iteration[1]:.6f}\t{iteration[2]:.6e}")

    # Final results
    print(f"\nRoot found: {root:.6f}")
    print(f"Function value at root: {function_value:.6e}")
    print(f"Absolute error: {absolute_error:.6e}")

except ValueError as e:
    print("Error:", e)
