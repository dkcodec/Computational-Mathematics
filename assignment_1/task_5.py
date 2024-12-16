import numpy as np
import matplotlib.pyplot as plt

# Define the function
f = lambda x: x**2 - 2

# False Position Method
def false_position_method(func, a, b, tolerance=1e-6, max_iterations=100):
    """
    False Position method for finding the root of a function.

    Parameters:
        func: The function for which to find the root.
        a, b: Initial interval [a, b].
        tolerance: Convergence tolerance.
        max_iterations: Maximum number of iterations.

    Returns:
        root: The found root.
        iterations: List of iteration details (iteration number, root approximation, absolute error, relative error).
    """
    if func(a) * func(b) >= 0:
        raise ValueError("The function must have opposite signs at the endpoints a and b.")

    iterations = []
    for i in range(max_iterations):
        # Compute the root approximation using the False Position formula
        c = b - (func(b) * (b - a)) / (func(b) - func(a))

        # Compute errors
        absolute_error = abs(func(c))
        relative_error = abs((b - a) / c) if c != 0 else float('inf')

        # Record iteration details
        iterations.append((i + 1, c, absolute_error, relative_error))

        # Check for convergence
        if absolute_error < tolerance:
            return c, iterations

        # Update interval
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c

    raise ValueError("False Position method did not converge within the maximum number of iterations.")

# Initial interval
a, b = 0, 2

# Find root and collect iteration details
try:
    root, iteration_details = false_position_method(f, a, b)

    # Create a table of iterations
    print("Iteration\tRoot Approximation\tAbsolute Error\tRelative Error")
    for iteration in iteration_details:
        print(f"{iteration[0]}\t{iteration[1]:.6f}\t{iteration[2]:.6e}\t{iteration[3]:.6e}")

    # Extract data for plotting
    iteration_numbers = [item[0] for item in iteration_details]
    absolute_errors = [item[2] for item in iteration_details]

    # Plot convergence graph
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_numbers, absolute_errors, marker='o', label='Absolute Error')
    plt.yscale('log')
    plt.title("Convergence of False Position Method")
    plt.xlabel("Iteration Number")
    plt.ylabel("Absolute Error (log scale)")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    # Final results
    print(f"\nRoot found: {root:.6f}")
    print(f"Function value at root: {f(root):.6e}")

except ValueError as e:
    print("Error:", e)

# Explanation of convergence
print("\nExplanation:")
print("The False Position method converges more slowly than the Newton-Raphson method because it does not update both endpoints simultaneously. It relies on linear interpolation and keeps one endpoint fixed, which can cause imbalance in interval reduction and slower convergence.")
