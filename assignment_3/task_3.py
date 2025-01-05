import numpy as np

def power_method(A, v0, tol, max_iter):
    """Power iteration method for the largest eigenvalue."""
    v = v0 / np.linalg.norm(v0)  # Normalize the initial vector
    for _ in range(max_iter):
        w = np.dot(A, v)  # Matrix-vector multiplication
        lambda_new = np.dot(w, v)  # Eigenvalue approximation
        v_new = w / np.linalg.norm(w)  # Normalize the new vector

        # Convergence check
        if np.linalg.norm(v_new - v) < tol:
            return lambda_new, v_new

        v = v_new
    return lambda_new, v_new

# Example matrix
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=float)
v0 = np.array([1, 0, 0], dtype=float)

# Finding the eigenvalue and eigenvector
lambda_max, eigenvector = power_method(A, v0, tol=1e-6, max_iter=100)

# Printing results
print("\nMatrix A:")
print(A)

print("\nPower Method Results:")
print(f"Largest eigenvalue: {np.round(lambda_max, 6)}")
print("Corresponding eigenvector:")
print(np.round(eigenvector, 6))

# Verification: A * eigenvector ≈ lambda_max * eigenvector
verification = np.dot(A, eigenvector)
scaled_vector = lambda_max * eigenvector

print("\nVerification (A * eigenvector ≈ lambda_max * eigenvector):")
print("A * eigenvector:")
print(np.round(verification, 6))
print("lambda_max * eigenvector:")
print(np.round(scaled_vector, 6))
