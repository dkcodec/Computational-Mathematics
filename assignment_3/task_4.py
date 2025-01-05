import numpy as np

A = np.array([
    [4, 1, 2, 0],
    [1, 3, 1, 2],
    [2, 1, 5, 1],
    [0, 2, 1, 4]
], dtype=float)


def givens_qr(A):
    """
    Perform QR factorization using Givens rotations.
    Returns Q, R such that A = Q * R.
    """
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    # We will zero out below the main diagonal column-by-column
    for j in range(n):
        for i in range(j+1, m):
            if R[i, j] != 0:
                # Compute the rotation to zero out R[i,j]
                r_ji = R[j, j]
                r_ij = R[i, j]

                # Hypotenuse
                r = np.hypot(r_ji, r_ij)
                c = r_ji / r
                s = -r_ij / r

                # Construct Givens rotation G
                G = np.eye(m)
                G[j, j] = c
                G[j, i] = -s
                G[i, j] = s
                G[i, i] = c

                # Apply G to R and accumulate G in Q
                R = G @ R
                Q = Q @ G.T  # Because final Q = (G1 * G2 * ...).T, 
                             # but we accumulate in the forward direction
    return Q, R

# --- Givens QR on A ---
Q_givens, R_givens = givens_qr(A)

print("Givens QR Factorization:")
print("Q (Givens):")
print(np.round(Q_givens, 6))
print("\nR (Givens):")
print(np.round(R_givens, 6))

# Verification: Q * R ≈ A
reconstructed_A_givens = Q_givens @ R_givens
print("\nCheck: Q_givens * R_givens ≈ A")
print(np.round(reconstructed_A_givens, 6))



def householder_qr(A):
    """
    Perform QR factorization using Householder reflections.
    Returns Q, R such that A = Q * R.
    """
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)

    for k in range(n):
        # Vector x = k-th column of R from row k to the end
        x = R[k:, k]
        if np.allclose(x, 0):
            continue  # Skip if all zeros

        # Norm of x
        norm_x = np.linalg.norm(x)
        # Sign ensures a stable reflection
        alpha = -np.sign(x[0]) * norm_x
        # Householder vector u
        u = x.copy()
        u[0] -= alpha

        # Reflection normalization
        norm_u = np.linalg.norm(u)
        if np.isclose(norm_u, 0):
            continue

        u = u / norm_u  # Make u a unit vector
        # Construct the reflection block for rows k..
        Hk = np.eye(m - k) - 2.0 * np.outer(u, u)

        # Apply to R (only the bottom-right portion)
        R[k:, k:] = Hk @ R[k:, k:]
        # Construct full-size Householder matrix
        H = np.eye(m)
        H[k:, k:] = Hk
        # Accumulate in Q
        Q = Q @ H
    
    return Q, R

# --- Householder QR on A ---
Q_house, R_house = householder_qr(A)

print("\nHouseholder QR Factorization:")
print("Q (Householder):")
print(np.round(Q_house, 6))
print("\nR (Householder):")
print(np.round(R_house, 6))

# Verification: Q * R ≈ A
reconstructed_A_house = Q_house @ R_house
print("\nCheck: Q_house * R_house ≈ A")
print(np.round(reconstructed_A_house, 6))
