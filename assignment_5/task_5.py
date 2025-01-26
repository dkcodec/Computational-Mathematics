import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given data
x = np.array([0, 2, 5, 8], dtype=float)
y = np.array([4, 8, 14, 25], dtype=float)
x_to_estimate = 3  # Point to estimate f(x)

# Step 1: Calculate the divided difference table
n = len(x)
divided_diff_table = np.zeros((n, n))
divided_diff_table[:, 0] = y

for j in range(1, n):
    for i in range(n - j):
        divided_diff_table[i, j] = (divided_diff_table[i + 1, j - 1] - divided_diff_table[i, j - 1]) / (x[i + j] - x[i])

# Convert to a DataFrame for better visualization
divided_diff_df = pd.DataFrame(divided_diff_table)
divided_diff_df.columns = [f"Div^{j}y" for j in range(n)]

# Step 2: Newton's Divided Difference Formula
# Initialize the estimated value with the first term
interpolated_value = divided_diff_table[0, 0]
product_term = 1

# Add successive terms from the divided difference table
for i in range(1, n):
    product_term *= (x_to_estimate - x[i - 1])
    interpolated_value += product_term * divided_diff_table[0, i]

# Visualization: Plot the data points and the interpolated value
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Given Data")
plt.scatter([x_to_estimate], [interpolated_value], color='red', label=f"Estimated f({x_to_estimate}) = {interpolated_value:.2f}", zorder=5)
plt.title("Newton's Divided Difference Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Display the results
print("Divided Difference Table:")
print(divided_diff_df)
print(f"Estimated f({x_to_estimate}) using Newton's Divided Difference Formula = {interpolated_value:.2f}")
