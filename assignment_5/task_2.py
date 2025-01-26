import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data
x = np.array([3, 4, 5, 6, 7], dtype=float)
y = np.array([2.2, 3.5, 5.1, 7.3, 10.0], dtype=float)
x_to_estimate = 5.5  # Point to estimate f(x)

# Step 1: Calculate the backward differences table
n = len(x)
backward_diff_table = np.zeros((n, n))
backward_diff_table[:, 0] = y

for j in range(1, n):
    for i in range(n - 1, j - 2, -1):
        backward_diff_table[i, j] = backward_diff_table[i, j - 1] - backward_diff_table[i - 1, j - 1]

# Convert to a DataFrame for better visualization
backward_diff_df = pd.DataFrame(backward_diff_table)
backward_diff_df.columns = [f"∇^{j}y" for j in range(n)]

# Step 2: Newton's Backward Interpolation Formula
h = x[1] - x[0]  # Uniform step size
u = (x_to_estimate - x[-1]) / h  # u value for the formula

# Calculate f(x) using Newton's Backward Formula
interpolated_value = y[-1]
u_term = 1  # To calculate u(u+1)(u+2)...
for i in range(1, n):
    u_term *= (u + (i - 1))
    interpolated_value += (u_term * backward_diff_table[n - 1, i]) / np.math.factorial(i)

# Visualization: Plot the data points and the interpolated value
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Given Data")
plt.scatter([x_to_estimate], [interpolated_value], color='red', label=f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}", zorder=5)
plt.title("Newton's Backward Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Display the results
print("Backward Difference Table:")
print(backward_diff_df)
print(f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}")

