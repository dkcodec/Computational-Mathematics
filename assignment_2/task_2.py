import numpy as np
from prettytable import PrettyTable

# Coefficient matrix A
A = np.array([
    [2, 3, 1],
    [4, 11, -1],
    [-2, 1, 7]
], dtype=float)

# Right-hand side vector b
b = np.array([10, 33, 15], dtype=float)

# Form the augmented matrix [A|b]
augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

def gaussian_elimination(A, b):
    n = len(b)
    # Form the augmented matrix
    aug = np.hstack((A, b.reshape(-1, 1)))
    
    print("Initial Augmented Matrix:")
    print_matrix(aug)
    
    # Forward Elimination with Partial Pivoting
    for i in range(n):
        # Partial Pivoting: Find the maximum element in the current column
        max_row = i + np.argmax(np.abs(aug[i:, i]))
        if aug[max_row, i] == 0:
            raise ValueError("Matrix is singular!")
        
        # Swap the current row with the max_row
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
            print(f"\nSwapped Row {i+1} with Row {max_row+1}:")
            print_matrix(aug)
        
        # Eliminate entries below the pivot
        for j in range(i+1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] = aug[j, i:] - factor * aug[i, i:]
            print(f"\nEliminating variable x{i+1} from Row {j+1}:")
            print_matrix(aug)
    
    # Extract the upper triangular matrix
    upper_triangular = aug[:, :-1]
    constants = aug[:, -1]
    
    return upper_triangular, constants

def back_substitution(U, c):
    n = len(c)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        if U[i, i] == 0:
            raise ValueError("Matrix is singular!")
        x[i] = (c[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        print(f"Solving for x{i+1}: {x[i]:.6f}")
    
    return x

def print_matrix(matrix):
    n_rows, n_cols = matrix.shape
    table = PrettyTable()
    # Generate column names
    column_names = [f"x{i+1}" for i in range(n_cols -1)] + ["b"]
    table.field_names = column_names
    # Add rows to the table
    for row in matrix:
        formatted_row = [f"{elem:.6f}" for elem in row]
        table.add_row(formatted_row)
    print(table)


# Perform Gaussian Elimination
upper_tri, constants = gaussian_elimination(A, b)

print("\nUpper Triangular Matrix [U|c]:")
print_matrix(np.hstack((upper_tri, constants.reshape(-1,1))))

# Perform Back Substitution
print("\nBack Substitution Steps:")
solution = back_substitution(upper_tri, constants)

# Display the solution
print("\nSolution Vector:")
for idx, val in enumerate(solution, start=1):
    print(f"x{idx} = {val:.6f}")

