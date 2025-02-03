import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 2, 4, 6, 8])
y1 = np.array([1, 4, 16, 36, 64])
x_target1 = 0

def newton_forward_difference_second_derivative(x, y, x_target):
    n = len(y)
    h = x[1] - x[0]
    forward_diff = np.zeros((n, n))
    forward_diff[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            forward_diff[i, j] = forward_diff[i + 1, j - 1] - forward_diff[i, j - 1]
    return (forward_diff[0, 2] / (h ** 2)) - (forward_diff[0, 3] / (h ** 2))

result2 = newton_forward_difference_second_derivative(x1, y1, x_target1)
print("Task 2 Result:", result2)