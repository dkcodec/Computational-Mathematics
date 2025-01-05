import numpy as np

def lu_factorization(A):
    """LU factorization of a matrix."""
    n = A.shape[0]
    L = np.eye(n)  # Lower triangular matrix
    U = A.copy()  # Upper triangular matrix

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    return L, U

def solve_lu(L, U, b):
    """Solving the system using LU decomposition."""
    # Forward substitution for Ly = b
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Back substitution for Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# Example of a matrix and right-hand side
A = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1]  ,[0, 3, -1, 8]], dtype=float)
b = np.array([5, 20, -10, 15], dtype=float)

# LU factorization
L, U = lu_factorization(A)

# Printing results
print("\nMatrix A:")
print(A)

print("\nLower triangular matrix L:")
print(np.round(L, 6))

print("\nUpper triangular matrix U:")
print(np.round(U, 6))

# Solve the system Ax = b
x = solve_lu(L, U, b)

print("\nSolution of Ax = b:")
print(np.round(x, 6))

# Verification: A * x ≈ b
check = np.dot(A, x)

print("\nVerification (A * x ≈ b):")
print("b (original):", np.round(b, 6))
print("b (computed):", np.round(check, 6))
