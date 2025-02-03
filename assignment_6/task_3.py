import numpy as np
import matplotlib.pyplot as plt

# Task 3: First Derivative Using Newton’s Backward Difference Formula
def newton_backward_difference_first_derivative(x, y, x_target):
    n = len(y)
    h = x[1] - x[0]
    backward_diff = np.zeros((n, n))
    backward_diff[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            backward_diff[i, j] = backward_diff[i, j - 1] - backward_diff[i - 1, j - 1]
    return (backward_diff[n - 1, 1] / h) - (backward_diff[n - 1, 2] / (2 * h)) + (backward_diff[n - 1, 3] / (3 * h))

x3 = np.array([5, 6, 7, 8, 9])
y3 = np.array([10, 16, 26, 40, 58])
x_target3 = 9
result3 = newton_backward_difference_first_derivative(x3, y3, x_target3)
print("Task 3 Result:", result3)

plt.plot(x3, y3, marker='o', label='Data Points')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Task 3: Backward Difference")
plt.legend()
plt.show()