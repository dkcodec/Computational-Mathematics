import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([2, 5, 8, 10], dtype=float)
y = np.array([1.4, 2.3, 3.8, 4.6], dtype=float)
x_to_estimate = 6  # Point to estimate f(x)

# Step 1: Lagrange's Interpolation Formula
def lagrange_interpolation(x, y, x_to_estimate):
    n = len(x)
    estimated_value = 0
    for i in range(n):
        # Calculate the Lagrange basis polynomial L_i(x)
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x_to_estimate - x[j]) / (x[i] - x[j])
        # Add the contribution of L_i(x) * y_i to the total estimate
        estimated_value += L_i * y[i]
    return estimated_value

# Calculate the interpolated value
interpolated_value = lagrange_interpolation(x, y, x_to_estimate)

# Visualization: Plot the data points and the interpolated value
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Given Data")
plt.scatter([x_to_estimate], [interpolated_value], color='red', label=f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}", zorder=5)
plt.title("Lagrange's Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Display the result
print(f"Estimated f({x_to_estimate}) using Lagrange's Interpolation = {interpolated_value:.2f}")
