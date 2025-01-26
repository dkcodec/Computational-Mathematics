import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([1, 2.7, 5.8, 10.4, 16.5], dtype=float)
x_to_estimate = 2.5  # Point to estimate f(x)

# Step 1: Calculate the forward differences table
n = len(x)
forward_diff_table = np.zeros((n, n))
forward_diff_table[:, 0] = y

for j in range(1, n):
    for i in range(n - j):
        forward_diff_table[i, j] = forward_diff_table[i + 1, j - 1] - forward_diff_table[i, j - 1]

# Convert to a DataFrame for better visualization
forward_diff_df = pd.DataFrame(forward_diff_table)
forward_diff_df.columns = [f"Δ^{j}y" for j in range(n)]

# Step 2: Newton's Forward Interpolation Formula
h = x[1] - x[0]  # Uniform step size
u = (x_to_estimate - x[0]) / h  # u value for the formula

# Calculate f(x) using Newton's Forward Formula
interpolated_value = y[0]
u_term = 1  # To calculate u(u-1)(u-2)...
for i in range(1, n):
    u_term *= (u - (i - 1))
    interpolated_value += (u_term * forward_diff_table[0, i]) / np.math.factorial(i)

# Visualization: Plot the data points and the interpolated value
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Given Data")
plt.scatter([x_to_estimate], [interpolated_value], color='red', label=f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}", zorder=5)
plt.title("Newton's Forward Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Display the results
print("Forward Difference Table:")
print(forward_diff_df)
print(f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}")
