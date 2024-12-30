import numpy as np

def gauss_jordan(A, b):
    n = len(b)
    # Form the augmented matrix
    aug = np.hstack((A.astype(float), b.reshape(-1,1).astype(float)))
    
    for i in range(n):
        # Partial Pivoting: Find the maximum element in the current column
        max_row = np.argmax(np.abs(aug[i:, i])) + i
        if aug[max_row, i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")
        
        # Swap the current row with the max_row
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
        
        # Normalize the pivot row
        aug[i] = aug[i] / aug[i, i]
        
        # Eliminate all other entries in the current column
        for j in range(n):
            if j != i:
                aug[j] = aug[j] - aug[j, i] * aug[i]
    
    # Extract the solution
    x = aug[:, -1]
    return x

# Define the coefficient matrix A and vector b
A = np.array([
    [1, 1, 1],
    [2, -3, 4],
    [3, 4, 5]
])

b = np.array([9, 13, 40])

# Solve the system
solution = gauss_jordan(A, b)
print("Solution:")
for idx, val in enumerate(solution, start=1):
    print(f"x{idx} = {val:.6f}")


# The Gauss-Jordan elimination method offers several advantages compared to the traditional Gaussian elimination method. Below are the key benefits:

# a. Direct Solution Without Back Substitution
# Gauss-Jordan simplifies the solution process by eliminating the need for back substitution, making it more straightforward, especially for small systems.

# b. Facilitates Finding the Inverse Matrix
# Gauss-Jordan is versatile and efficient for computing the inverse of a matrix alongside solving linear systems.

# c. Better Numerical Stability and Precision
# Gauss-Jordan offers improved numerical stability, leading to more accurate solutions.