# Exponential Curve Fitting using Non-Linear Least Squares Method
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the exponential function model
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Data points
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.5, 4.7, 8.8, 16.2, 30.3])

# Provide an initial guess for the parameters a and b
initial_guess = [1.0, 0.5]

# Perform the curve fitting using scipy's curve_fit
# popt contains the optimal values for a and b
# pcov contains the covariance matrix which can be used to estimate the standard deviation of the parameters
popt, pcov = curve_fit(exponential_model, x_data, y_data, p0=initial_guess)

# Extract the optimal parameters
a_opt, b_opt = popt

# Calculate the standard deviations of the parameters from the covariance matrix
perr = np.sqrt(np.diag(pcov))
a_err, b_err = perr

# Generate fitted y values using the optimal parameters
y_fitted = exponential_model(x_data, a_opt, b_opt)

# Calculate residuals (differences between observed and fitted values)
residuals = y_data - y_fitted

# Display intermediate results
print("Exponential Curve Fitting Results:")
print("---------------------------------")
print(f"Estimated Parameters:")
print(f"a = {a_opt:.4f} ± {a_err:.4f}")
print(f"b = {b_opt:.4f} ± {b_err:.4f}\n")

# Display a table of data points with fitted values and residuals
print("Data Points with Fitted Values and Residuals:")
print("------------------------------------------------")
print(f"{'x':>5} {'y_observed':>12} {'y_fitted':>12} {'Residual':>10}")
for xi, yi, yf, res in zip(x_data, y_data, y_fitted, residuals):
    print(f"{xi:5} {yi:12.2f} {yf:12.2f} {res:10.2f}")

# Plotting the original data points and the fitted exponential curve
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='blue', label='Data Points')

# Generate a smooth x-axis for plotting the fitted curve
x_smooth = np.linspace(min(x_data), max(x_data), 100)
y_smooth = exponential_model(x_smooth, a_opt, b_opt)
plt.plot(x_smooth, y_smooth, color='red', label=f'Fitted Curve: y = {a_opt:.2f}e^({b_opt:.2f}x)')

# Adding titles and labels
plt.title('Exponential Curve Fitting using Non-Linear Least Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
