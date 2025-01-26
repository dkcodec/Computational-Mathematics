import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Given data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.3, 3.1, 4.9, 6.5, 8.1], dtype=float)
points_to_estimate = [2.5, 4.3]  # Points to estimate f(x)

# Step 1: Perform Cubic Spline Interpolation
cubic_spline = CubicSpline(x, y)

# Step 2: Estimate values for the given points
estimated_values = [cubic_spline(pt) for pt in points_to_estimate]

# Step 3: Visualization of the interpolation and estimated points
x_dense = np.linspace(min(x), max(x), 500)  # Dense points for smooth curve
y_dense = cubic_spline(x_dense)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'bo-', label="Given Data")
plt.plot(x_dense, y_dense, 'g-', label="Cubic Spline Interpolation")
for pt, est in zip(points_to_estimate, estimated_values):
    plt.scatter(pt, est, color='red', label=f"Estimated f({pt}) = {est:.2f}", zorder=5)
plt.title("Cubic Spline Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.show()

# Display the results
for pt, est in zip(points_to_estimate, estimated_values):
    print(f"Estimated f({pt}) using Cubic Spline Interpolation = {est:.2f}")
