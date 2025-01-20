# Fitting the model y = a + b*ln(x) + c*x^2 using Method of Least Squares
import numpy as np
import matplotlib.pyplot as plt

# Data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.6, 6.3, 11.5, 18.9])

# Number of data points
N = len(x)

# Constructing the design matrix X
# Column 1: 1 (for constant term a)
# Column 2: ln(x) (for coefficient b)
# Column 3: x^2 (for coefficient c)
X = np.column_stack((np.ones(N), np.log(x), x**2))

# Display the design matrix
print("Design Matrix (X):")
print(X)

# Computing the transpose of X
X_transpose = X.T

# Computing X^T * X
XtX = np.dot(X_transpose, X)

# Computing X^T * y
Xty = np.dot(X_transpose, y)

# Display intermediate summations
print("\nIntermediate Computations:")
print("X^T * X:")
print(XtX)
print("\nX^T * y:")
print(Xty)

# Solving the normal equations for theta = [a, b, c]^T
theta = np.linalg.solve(XtX, Xty)

# Extracting the constants a, b, c
a, b, c = theta

# Display the calculated constants
print("\nCalculated Constants:")
print(f"a (constant term) = {a:.4f}")
print(f"b (coefficient of ln(x)) = {b:.4f}")
print(f"c (coefficient of x^2) = {c:.4f}")

# Generate fitted y values using the calculated model
y_fitted = a + b * np.log(x) + c * x**2

# Display a table of data points with fitted values and residuals
print("\nData Points with Fitted Values and Residuals:")
print("------------------------------------------------")
print(f"{'x':>5} {'y_observed':>12} {'y_fitted':>12} {'Residual':>10}")
for xi, yi, yf in zip(x, y, y_fitted):
    residual = yi - yf
    print(f"{xi:5} {yi:12.2f} {yf:12.2f} {residual:10.2f}")

# Plotting the original data points and the fitted curve
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')

# To plot a smooth curve, generate more x values within the range
x_smooth = np.linspace(min(x), max(x), 100)
y_smooth = a + b * np.log(x_smooth) + c * x_smooth**2
plt.plot(x_smooth, y_smooth, color='red', label=f'Fitted Curve: y = {a:.2f} + {b:.2f}ln(x) + {c:.2f}x²')

# Adding title and labels
plt.title('Curve Fitting: y = a + b*ln(x) + c*x²')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
