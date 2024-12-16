import numpy as np
import matplotlib.pyplot as plt

# Define the function
f = lambda x: x**3 - 2*x**2 - 5

def custom_root_finder(func, x0, tolerance=1e-6, max_iterations=100):
    """
    Custom function to find the root of a given function using the Newton-Raphson method.
    
    Parameters:
    func: Function for which the root is to be found.
    x0: Initial guess for the root.
    tolerance: Convergence tolerance.
    max_iterations: Maximum number of iterations.

    Returns:
    Root of the function.
    """
    # Derivative of the function (numerical approximation)
    derivative = lambda x: (func(x + 1e-6) - func(x)) / 1e-6

    x = x0
    for iteration in range(max_iterations):
        f_value = func(x)
        f_derivative = derivative(x)

        if abs(f_value) < tolerance:
            return x

        if f_derivative == 0:
            raise ValueError("Derivative is zero. Newton-Raphson method fails.")

        # Update x using Newton-Raphson formula
        x = x - f_value / f_derivative

    raise ValueError("Root finding did not converge within the maximum number of iterations.")

# Generate values for x within the range [1, 4]
x_values = np.linspace(1, 4, 500)
y_values = f(x_values)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x) = x^3 - 2x^2 - 5')
plt.axhline(0, color='black', linestyle='--', label='y=0 (x-axis)')
plt.title("Graph of f(x) = x^3 - 2x^2 - 5")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()
plt.show()

# Approximate root visually from the graph ~ 2.7
approx_root = 2.7

# Calculate f(x) for the approximate root
f_approx_root = f(approx_root)

# Use custom function to find the exact root
exact_root = custom_root_finder(f, approx_root)

# Calculate the absolute error
absolute_error = abs(exact_root - approx_root)

# Print results
print("Approximate root (from graph):", approx_root)
print("Value of f(x) at approximate root:", f_approx_root)
print("Exact root (using custom method):", exact_root)
print("Absolute error:", absolute_error)
