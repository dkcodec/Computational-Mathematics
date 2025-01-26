import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data
x = np.array([10, 12, 14, 16, 18], dtype=float)
y = np.array([100, 144, 196, 256, 324], dtype=float)
x_to_estimate = 13  # Point to estimate f(x)

# Step 1: Calculate the central differences table
n = len(x)
central_diff_table = np.zeros((n, n))
central_diff_table[:, 0] = y

for j in range(1, n):
    for i in range(n - j):
        central_diff_table[i, j] = central_diff_table[i + 1, j - 1] - central_diff_table[i, j - 1]

# Convert to a DataFrame for better visualization
central_diff_df = pd.DataFrame(central_diff_table)
central_diff_df.columns = [f"Δ^{j}y" for j in range(n)]

# Step 2: Central Difference Interpolation Formula
h = x[1] - x[0]  # Uniform step size
u = (x_to_estimate - x[len(x) // 2]) / h  # u value for the formula

# Calculate f(x) using Central Difference Interpolation Formula
interpolated_value = y[len(x) // 2]
u_term = 1  # To calculate u(u^2 - 1)(u^2 - 4)...
factorial = 1
for i in range(1, n):
    if i % 2 == 1:  # Odd terms: forward differences
        u_term *= u
    else:  # Even terms: central differences
        factorial *= i
        u_term *= (u**2 - (i // 2)**2)
    if len(x) // 2 - (i // 2) >= 0:  # Ensure valid index for central differences
        interpolated_value += (u_term * central_diff_table[len(x) // 2 - (i // 2), i]) / factorial

# Visualization: Plot the data points and the interpolated value
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Given Data")
plt.scatter([x_to_estimate], [interpolated_value], color='red', label=f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}", zorder=5)
plt.title("Central Difference Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Display the results
print("Central Difference Table:")
print(central_diff_df)
print(f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}")

