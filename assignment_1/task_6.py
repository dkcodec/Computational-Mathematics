import numpy as np
import matplotlib.pyplot as plt

# Define the function
f = lambda x: x**2 - 6*x + 5

def g1(x):
    """
    Transformation of the equation into the form x = g(x)
    """
    return (x**2 + 5) / 6

# Iteration Method
def iteration_method(g, x0, true_root, num_iterations=10):
    """
    Iteration method for solving equations.

    Parameters:
        g: Transformation function g(x).
        x0: Initial guess.
        true_root: True root for error calculation.
        num_iterations: Number of iterations to perform.

    Returns:
        results: List of iteration details (iteration number, current guess, absolute error).
    """
    results = []
    x = x0

    for i in range(1, num_iterations + 1):
        next_x = g(x)
        absolute_error = abs(next_x - true_root)
        results.append((i, next_x, absolute_error))
        x = next_x

    return results

# Initial guess and true root
x0 = 0.5
true_root = 1  # One of the roots of the equation

# Perform iterations using g(x)
iteration_details = iteration_method(g1, x0, true_root)

# Display results
print("Iteration\tCurrent Guess\tAbsolute Error")
for iteration in iteration_details:
    print(f"\t{iteration[0]}\t{iteration[1]:.6f}\t{iteration[2]:.6e}")

# Extract data for plotting
iteration_numbers = [item[0] for item in iteration_details]
absolute_errors = [item[2] for item in iteration_details]

# Plot convergence graph
plt.figure(figsize=(10, 6))
plt.plot(iteration_numbers, absolute_errors, marker='o', label='Absolute Error')
plt.yscale('log')
plt.title("Convergence of Iteration Method")
plt.xlabel("Iteration Number")
plt.ylabel("Absolute Error (log scale)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
