import numpy as np

# Define the coefficient matrix A and vector b
A = np.array([
    [1.001, 0.999],
    [1.002, 1.000]
], dtype=float)

b = np.array([2, 2.001], dtype=float)

# Compute the determinant to check for ill-conditioning
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")

# Solve the system using numpy's linear solver
try:
    solution = np.linalg.solve(A, b)
    print("Numerical Solution:")
    print(f"x1 = {solution[0]:.6f}")
    print(f"x2 = {solution[1]:.6f}")
except np.linalg.LinAlgError as e:
    print("Error:", e)

# A system of linear equations is said to be ill-conditioned if its condition number is high. The condition number (κ(A)) of a matrix A is a measure of how much the output value of the function can change for a small change in the input argument. For linear systems, it quantifies the sensitivity of the solution vector x to changes or errors in the input data (coefficients and constants).