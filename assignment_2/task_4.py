import numpy as np
from prettytable import PrettyTable


#Convergence: The method converges if the coefficient matrix satisfies certain conditions (e.g., diagonal dominance).
#Usage of Latest Values: Unlike the Jacobi method, Gauss-Seidel uses updated values immediately within an iteration, potentially leading to faster convergence.
#Memory Efficiency: Gauss-Seidel updates the solution vector in-place, reducing memory requirements compared to Jacobi.

# Coefficient matrix A
A = np.array([
    [8, -3, 2],
    [4, 11, -1],
    [6, 3, 12]
], dtype=float)

# Right-hand side vector b
b = np.array([20, 33, 36], dtype=float)


def gauss_seidel(A, b, x0, tol=1e-5, max_iterations=1000):
    """
    Solves the linear system Ax = b using the Gauss-Seidel iterative method.

    Parameters:
    A (ndarray): Coefficient matrix.
    b (ndarray): Right-hand side vector.
    x0 (ndarray): Initial guess vector.
    tol (float): Tolerance for convergence.
    max_iterations (int): Maximum number of iterations.

    Returns:
    x (ndarray): Solution vector.
    iterations (int): Number of iterations performed.
    history (list): History of solutions per iteration.
    """
    n = len(b)
    x = x0.copy()
    history = [x.copy()]
    
    for k in range(1, max_iterations + 1):
        x_new = np.copy(x)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Using updated values
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Using old values
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Calculate the maximum difference
        diff = np.abs(x_new - x)
        history.append(x_new.copy())
        
        # Check for convergence
        if np.max(diff) < tol:
            return x_new, k, history
        
        x = x_new.copy()
    
    # If no convergence within max_iterations
    return x, max_iterations, history


# Initial guess
x0 = np.array([0, 0, 0], dtype=float)

# Run Gauss-Seidel Method
solution, iterations, history = gauss_seidel(A, b, x0)

# Prepare the table
table = PrettyTable()
table.field_names = ["Iteration", "x1", "x2", "x3"]

for idx, vec in enumerate(history):
    table.add_row([idx, f"{vec[0]:.6f}", f"{vec[1]:.6f}", f"{vec[2]:.6f}"])

print(table)
print(f"\nConverged in {iterations} iterations.\n")

# Display the solution
print("Solution Vector:")
print(f"x1 = {solution[0]:.6f}")
print(f"x2 = {solution[1]:.6f}")
print(f"x3 = {solution[2]:.6f}")

#Stopping Criterion Impact:

# Precision vs. Execution Time: A stricter tolerance (10^-5) ensures higher accuracy but may require more iterations, increasing execution time. Conversely, a looser tolerance reduces execution time but may compromise solution accuracy.
# Optimal Balance: Selecting an appropriate tolerance is essential to balance computational efficiency with the required solution precision based on application needs.