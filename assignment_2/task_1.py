import numpy as np
from prettytable import PrettyTable

# Coefficient matrix A
A = np.array([
    [10, -1, -2],
    [-2, 10, -1],
    [-1, -2, 10]
], dtype=float)

# Right-hand side vector b
b = np.array([5, -6, 15], dtype=float)


def is_strictly_diagonally_dominant(A):
    D = np.abs(np.diag(A))
    S = np.sum(np.abs(A), axis=1) - D
    return np.all(D > S)

# Check for diagonal dominance
if is_strictly_diagonally_dominant(A):
    print("The matrix A is strictly diagonally dominant.\n")
else:
    print("The matrix A is NOT strictly diagonally dominant.\n")


def jacobi_method(A, b, x0, tol=1e-6, max_iterations=100):
    n = len(b)
    x_old = x0.copy()
    x_new = np.zeros_like(x0)
    
    # Prepare a table to display iterations
    table = PrettyTable()
    table.field_names = ["Iteration", "x1", "x2", "x3"]
    table.add_row([0, f"{x_old[0]:.6f}", f"{x_old[1]:.6f}", f"{x_old[2]:.6f}"])
    
    for k in range(1, max_iterations + 1):
        for i in range(n):
            s = 0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x_old[j]
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Add current iteration to the table
        table.add_row([k, f"{x_new[0]:.6f}", f"{x_new[1]:.6f}", f"{x_new[2]:.6f}"])
        
        # Check for convergence
        if np.all(np.abs(x_new - x_old) < tol):
            print(table)
            print(f"\nConverged in {k} iterations.")
            return x_new
        
        x_old = x_new.copy()
    
    print(table)
    print(f"\nDid not converge within {max_iterations} iterations.")
    return x_new

# A: Coefficient matrix.
# b: Right-hand side vector.
# x0: Initial guess vector.
# tol: Tolerance for convergence.
# max_iterations: Maximum number of iterations to prevent infinite loops.

# Initial guess
x0 = np.array([0, 0, 0], dtype=float)

# Run Jacobi Method
solution = jacobi_method(A, b, x0)

# Display the solution
print("\nSolution Vector:")
print(f"x1 = {solution[0]:.6f}")
print(f"x2 = {solution[1]:.6f}")
print(f"x3 = {solution[2]:.6f}")

# Diagonal Dominance: The coefficient matrix is strictly diagonally dominant, satisfying the convergence criterion for the Jacobi method.
# Convergence Dependence: The structure of the system, particularly strict diagonal dominance, ensures the convergence of the Jacobi method. The method's effectiveness is influenced by the matrix's diagonal dominance, spectral radius, and condition number.