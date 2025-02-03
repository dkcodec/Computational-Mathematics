import numpy as np
import matplotlib.pyplot as plt

# Task 4: First Derivative Using Lagrange’s Interpolation Formula
def lagrange_interpolation_derivative(x, y, x_target):
    n = len(x)
    derivative = 0
    for i in range(n):
        term = 0
        for j in range(n):
            if i != j:
                num = 1
                den = 1
                for k in range(n):
                    if k != i and k != j:
                        num *= (x_target - x[k])
                        den *= (x[i] - x[k])
                term += (y[i] - y[j]) / (x[i] - x[j]) * num / den
        derivative += term
    return derivative

x4 = np.array([1, 2, 4, 7])
y4 = np.array([3, 6, 12, 21])
x_target4 = 3
result4 = lagrange_interpolation_derivative(x4, y4, x_target4)
print("Task 4 Result:", result4)