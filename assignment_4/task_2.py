# Polynomial Curve Fitting using Method of Least Squares (Quadratic Fit)
import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([2, 3, 6, 11, 18])

# Number of data points
N = len(x)

# Calculating necessary summations
S_x = np.sum(x)
S_x2 = np.sum(x**2)
S_x3 = np.sum(x**3)
S_x4 = np.sum(x**4)
S_y = np.sum(y)
S_xy = np.sum(x * y)
S_x2y = np.sum((x**2) * y)

# Display intermediate summations
print("Sum of x (S_x):", S_x)
print("Sum of x^2 (S_x2):", S_x2)
print("Sum of x^3 (S_x3):", S_x3)
print("Sum of x^4 (S_x4):", S_x4)
print("Sum of y (S_y):", S_y)
print("Sum of x*y (S_xy):", S_xy)
print("Sum of x^2*y (S_x2y):", S_x2y)

# Setting up the normal equations matrix
# | S_x4  S_x3  S_x2 |   | a |   | S_x2y |
# | S_x3  S_x2  S_x   | * | b | = | S_xy  |
# | S_x2  S_x   N     |   | c |   | S_y   |

# Constructing the matrix of coefficients
A = np.array([
    [S_x4, S_x3, S_x2],
    [S_x3, S_x2, S_x],
    [S_x2, S_x,  N ]
])

# Constructing the right-hand side vector
B = np.array([S_x2y, S_xy, S_y])

# Solving for [a, b, c] using numpy's linear algebra solver
coefficients = np.linalg.solve(A, B)
a, b, c = coefficients

# Display the calculated coefficients
print("\nCalculated Coefficients:")
print(f"a (quadratic term) = {a}")
print(f"b (linear term) = {b}")
print(f"c (constant term) = {c}")

# Generate fitted y values using the quadratic model
y_fitted = a * x**2 + b * x + c

# Plotting the original data points
plt.scatter(x, y, color='blue', label='Data Points')

# Plotting the fitted quadratic curve
# To make the curve smooth, generate more x points within the range
x_fit = np.linspace(min(x), max(x), 100)
y_fit = a * x_fit**2 + b * x_fit + c
plt.plot(x_fit, y_fit, color='red', label=f'Fitted Curve: y = {a:.2f}x² + {b:.2f}x + {c:.2f}')

# Adding title and labels
plt.title('Quadratic Curve Fitting using Least Squares Method')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Display the plot grid
plt.grid(True)

# Show the plot
plt.show()
