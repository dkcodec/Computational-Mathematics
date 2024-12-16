import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
f = lambda x: x**2 - 3*x + 2
f_prime = lambda x: 2*x - 3

# Newton-Raphson Method
def newton_raphson_method(func, func_prime, x0, tolerance=1e-6, max_iterations=100):
    iterations = []
    for i in range(max_iterations):
        f_x = func(x0)
        f_prime_x = func_prime(x0)
        
        if f_prime_x == 0:
            raise ValueError("Derivative is zero. Newton-Raphson method fails.")

        x1 = x0 - f_x / f_prime_x

        # Record iteration details
        absolute_error = abs(x1 - x0)
        relative_error = abs(absolute_error / x1) if x1 != 0 else float('inf')
        iterations.append((i + 1, x0, x1, absolute_error, relative_error))

        if absolute_error < tolerance:
            break

        x0 = x1

    return x1, iterations

# Initial guess
x0 = 2.5

# Find root and collect iteration details
root, iteration_details = newton_raphson_method(f, f_prime, x0)

# Create a table of iterations
print("Iteration\tCurrent Guess\tNext Guess\tAbsolute Error\tRelative Error")
for iteration in iteration_details:
    print(f"\t{iteration[0]}\t{iteration[1]:.6f}\t{iteration[2]:.6f}\t{iteration[3]:.6e}\t{iteration[4]:.6e}")

# Extract data for plotting
iteration_numbers = [item[0] for item in iteration_details]
absolute_errors = [item[3] for item in iteration_details]

# Plot convergence graph
plt.figure(figsize=(10, 6))
plt.plot(iteration_numbers, absolute_errors, marker='o', label='Absolute Error')
plt.yscale('log')
plt.title("Convergence of Newton-Raphson Method")
plt.xlabel("Iteration Number")
plt.ylabel("Absolute Error (log scale)")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Final root result
print(f"\nRoot found: {root:.6f}")
print(f"Function value at root: {f(root):.6e}")
