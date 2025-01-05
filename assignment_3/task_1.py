import numpy as np

def iterative_inverse(A, B, tol, max_iter):
    """Iterative method for finding the inverse matrix."""
    n = A.shape[0]
    I = np.eye(n)  # Identity matrix
    for _ in range(max_iter):
        E = np.dot(A, B) - I  # Calculating the error
        B_new = B - np.dot(B, E)  # Update step
        # Checking if the desired accuracy is achieved
        if np.linalg.norm(E, ord='fro') < tol:
            return B_new
        B = B_new
    return B_new

# Given matrix
A = np.array([[5, 2, 1], [2, 6, 3], [1, 3, 7]], dtype=float)

# Initial approximation: B = (1 / tr(A)) * I
trace_A = np.trace(A)
B = (1 / trace_A) * np.eye(A.shape[0])

# Iterative process
tol = 1e-6
max_iter = 100
A_inv_iterative = iterative_inverse(A, B, tol=tol, max_iter=max_iter)

# Built-in numpy function for comparison
A_inv_builtin = np.linalg.inv(A)

# Printing results
print("\nMatrix A:")
print(A)

print("\nIterative inverse of A:")
print(np.round(A_inv_iterative, 6))

print("\nInverse of A using numpy.linalg.inv:")
print(np.round(A_inv_builtin, 6))

# Verifying the result: A * A_inv ≈ I
I_approx_iterative = np.dot(A, A_inv_iterative)
I_approx_builtin = np.dot(A, A_inv_builtin)

print("\nVerification (A * A_inv_iterative):")
print(np.round(I_approx_iterative, 6))

print("\nVerification (A * A_inv_builtin):")
print(np.round(I_approx_builtin, 6))
