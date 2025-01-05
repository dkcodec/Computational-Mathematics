import numpy as np

A = np.array([
    [1,   1,   0.5 ],
    [1,   1,   0.25],
    [0.5, 0.25, 2  ]
], dtype=float)


def jacobi_eigenvalues(A, tol=1e-6, max_iter=1000):
    """
    Finds all eigenvalues (and optionally eigenvectors) of a symmetric matrix A
    using the Jacobi method.
    
    Parameters:
    -----------
    A : np.ndarray (symmetric matrix)
    tol : float, tolerance
    max_iter : int, maximum number of iterations
    
    Returns:
    --------
    eigenvalues : np.ndarray
        The eigenvalues of A.
    """
    # Copy to avoid modifying original
    A = A.copy()
    n = A.shape[0]
    
    # We don't strictly need eigenvectors here, but let's keep track if needed:
    # Q = np.eye(n)
    
    for iteration in range(max_iter):
        # 1. Find the largest off-diagonal element in magnitude
        off_diag_max = 0.0
        p = 0
        q = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > off_diag_max:
                    off_diag_max = abs(A[i, j])
                    p, q = i, j
        
        # 2. Check stopping criterion
        if off_diag_max < tol:
            break
        
        # 3. Compute the Jacobi rotation
        # The angle theta satisfies: tan(2 theta) = 2 A[p, q] / (A[p, p] - A[q, q])
        if abs(A[p, p] - A[q, q]) < 1e-15:
            # This prevents division by zero and handles near-equal diagonal terms
            theta = np.pi / 4  # 45 degrees
        else:
            phi = (A[q, q] - A[p, p]) / (2 * A[p, q])
            theta = np.arctan(1.0 / phi)
        
        # 4. Compute cos and sin
        c = np.cos(theta)
        s = np.sin(theta)
        
        # 5. Rotate: A' = J^T A J
        # Update diagonal elements
        a_pp = A[p, p]
        a_qq = A[q, q]
        
        A[p, p] = c**2 * a_pp - 2*s*c*A[p, q] + s**2 * a_qq
        A[q, q] = s**2 * a_pp + 2*s*c*A[p, q] + c**2 * a_qq
        A[p, q] = 0.0  # By construction, this is zeroed
        A[q, p] = 0.0
        
        # Update off-diagonal elements
        for k in range(n):
            if k != p and k != q:
                a_pk = A[p, k]
                a_qk = A[q, k]
                A[p, k] = c*a_pk - s*a_qk
                A[k, p] = A[p, k]  # Symmetric
                A[q, k] = s*a_pk + c*a_qk
                A[k, q] = A[q, k]

        # (If tracking eigenvectors Q: multiply Q by the rotation as well)
        # J = np.eye(n)
        # J[p, p] =  c;  J[p, q] = -s
        # J[q, p] =  s;  J[q, q] =  c
        # Q = Q @ J
        
    # After convergence, the diagonal of A contains the eigenvalues
    eigenvalues = np.diag(A)
    return eigenvalues

# Use Jacobi's method
jacobi_vals = jacobi_eigenvalues(A, tol=1e-6, max_iter=1000)

print("Eigenvalues via Jacobi's method:")
print(np.round(np.sort(jacobi_vals), 6))

# Compare with NumPy's built-in function
numpy_vals = np.linalg.eigvals(A)
print("\nEigenvalues via np.linalg.eigvals:")
print(np.round(np.sort(numpy_vals), 6))
