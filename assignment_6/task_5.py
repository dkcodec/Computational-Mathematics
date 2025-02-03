import numpy as np
import matplotlib.pyplot as plt

# Task 5: Finding Maximum or Minimum in a Tabulated Function
def find_max_min(x, y):
    dy_dx = np.gradient(y, x)
    extrema_indices = np.where(np.isclose(dy_dx, 0, atol=1e-2))[0]
    if len(extrema_indices) == 0:
        return None, "No extremum found"
    second_derivative = np.gradient(dy_dx, x)
    extrema_points = []
    for i in extrema_indices:
        if second_derivative[i] > 0:
            extrema_points.append((x[i], y[i], "Minimum"))
        elif second_derivative[i] < 0:
            extrema_points.append((x[i], y[i], "Maximum"))
    return extrema_points

x5 = np.array([2, 4, 6, 8, 10])
y5 = np.array([5, 7, 8, 6, 3])
result5 = find_max_min(x5, y5)
print("Task 5 Result:", result5)

plt.plot(x5, y5, marker='o', label='Data Points')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Task 5: Extrema Detection")
plt.legend()
plt.show()