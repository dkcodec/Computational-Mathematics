# Linear Curve Fitting using Method of Least Squares
import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 8, 11, 15])

# Number of data points
N = len(x)

# Calculating the necessary summations
S_x = np.sum(x)
S_y = np.sum(y)
S_xy = np.sum(x * y)
S_xx = np.sum(x * x)

# Display intermediate results
print("Sum of x (S_x):", S_x)
print("Sum of y (S_y):", S_y)
print("Sum of x*y (S_xy):", S_xy)
print("Sum of x^2 (S_xx):", S_xx)

# Calculating slope (m) and intercept (c) using least squares formulas
m = (N * S_xy - S_x * S_y) / (N * S_xx - S_x**2)
c = (S_y - m * S_x) / N

# Display the calculated slope and intercept
print("\nCalculated Slope (m):", m)
print("Calculated Intercept (c):", c)

# Generate fitted y values
y_fitted = m * x + c

# Plotting the original data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plotting the fitted line
plt.plot(x, y_fitted, color='red', label=f'Fitted Line: y = {m:.2f}x + {c:.2f}')

# Adding title and labels
plt.title('Linear Curve Fitting using Least Squares Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
