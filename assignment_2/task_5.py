import numpy as np
from prettytable import PrettyTable
import time

# Coefficient matrix A
A = np.array([
    [5, 1, 1],
    [1, 4, 2],
    [1, 1, 5]
], dtype=float)

# Right-hand side vector b
b = np.array([10, 12, 15], dtype=float)

def sor_method(A, b, x0, omega, tol=1e-5, max_iterations=10000):
    """
    Solves Ax = b using the Relaxation(Successive Over-Relaxation (SOR)) method.

    Parameters:
    A (ndarray): Coefficient matrix.
    b (ndarray): Right-hand side vector.
    x0 (ndarray): Initial guess vector.
    omega (float): Relaxation parameter.
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
        x_new = x.copy()
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Using updated values
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Using old values
            x_i_old = x[i]
            x_new[i] = (1 - omega) * x_i_old + (omega / A[i, i]) * (b[i] - sum1 - sum2)
        
        # Calculate the maximum difference
        diff = np.abs(x_new - x)
        history.append(x_new.copy())
        
        # Check for convergence
        if np.max(diff) < tol:
            return x_new, k, history
        
        x = x_new.copy()
    
    # If no convergence within max_iterations
    return x, max_iterations, history


def print_iteration_table(history, omega):
    table = PrettyTable()
    table.field_names = ["Iteration", "x1", "x2", "x3"]
    
    for idx, vec in enumerate(history):
        table.add_row([idx, f"{vec[0]:.6f}", f"{vec[1]:.6f}", f"{vec[2]:.6f}"])
    
    print(f"\nRelaxation Parameter ω = {omega}")
    print(table)


# Initial guess
x0 = np.array([0, 0, 0], dtype=float)

# Define relaxation parameters
omegas = [1.1, 1.5]

# Dictionary to store results
results = {}

for omega in omegas:
    start_time = time.time()
    solution, iterations, history = sor_method(A, b, x0, omega)
    end_time = time.time()
    exec_time = end_time - start_time
    results[omega] = {
        "solution": solution,
        "iterations": iterations,
        "history": history,
        "time": exec_time
    }

# Print iteration tables
for omega in omegas:
    print_iteration_table(results[omega]["history"], omega)

# Print execution time and number of iterations
print("\nComparison of ω = 1.1 and ω = 1.5:")
comparison_table = PrettyTable()
comparison_table.field_names = ["ω", "Iterations", "Execution Time (s)"]
for omega in omegas:
    comparison_table.add_row([
        omega,
        results[omega]["iterations"],
        f"{results[omega]['time']:.6f}"
    ])
print(comparison_table)

# Print solutions
for omega in omegas:
    sol = results[omega]["solution"]
    print(f"\nSolution for ω = {omega}:")
    print(f"x1 = {sol[0]:.6f}")
    print(f"x2 = {sol[1]:.6f}")
    print(f"x3 = {sol[2]:.6f}")
