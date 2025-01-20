# Backward Difference Table Construction and Verification of Constant Second-Order Differences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backward_difference_table(x, y):
    """
    Constructs the backward difference table for given x and y data points.

    Parameters:
    x (list or np.array): The x-values.
    y (list or np.array): The y-values.

    Returns:
    pd.DataFrame: A pandas DataFrame representing the backward difference table.
    """
    n = len(y)
    # Initialize a list of lists to store differences
    diff_table = [y.copy()]
    
    # Compute backward differences up to the (n-1)th order
    for order in range(1, n):
        previous_diff = diff_table[-1]
        current_diff = []
        for i in range(len(previous_diff) - 1, 0, -1):
            delta = previous_diff[i] - previous_diff[i-1]
            current_diff.insert(0, delta)  # Insert at beginning to maintain order
        diff_table.append(current_diff)
    
    # Determine the maximum number of columns needed
    max_cols = n
    # Initialize a DataFrame with NaN values
    df = pd.DataFrame(np.nan, index=range(n), columns=range(max_cols))
    
    # Fill in the DataFrame with x and differences
    for i in range(n):
        df.iloc[i,0] = x[i]  # First column for x-values
        for j in range(1, min(i+1, max_cols)):
            df.iloc[i,j] = diff_table[j][i - j]
    
    # Rename columns for clarity
    column_names = ['x']
    for i in range(1, max_cols):
        column_names.append(f'Δ^{i}y')
    df.columns = column_names
    
    return df

# Data points
x = np.array([5, 6, 7, 8, 9])
y = np.array([1, 8, 27, 64, 125])

# Construct the backward difference table
diff_table_df = backward_difference_table(x, y)

# Display the backward difference table
print("Backward Difference Table:")
print(diff_table_df.to_string(index=False))

# Verify if the second-order differences are constant
# Extract the second-order differences
second_order_diffs = diff_table_df['Δ^2y'].dropna().values

# Check if all second-order differences are equal
if len(second_order_diffs) > 0:
    is_constant = np.allclose(second_order_diffs, second_order_diffs[0])
    print("\nSecond-order differences:", second_order_diffs)
    print("Are the second-order differences constant?", is_constant)
else:
    print("\nNot enough data to compute second-order differences.")

# Plot the backward differences (optional)
# Plotting second-order differences to visualize constancy
if len(second_order_diffs) > 0:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(second_order_diffs)+1), second_order_diffs, marker='o', linestyle='-', color='purple')
    plt.title('Second-Order Backward Differences')
    plt.xlabel('Order Index')
    plt.ylabel('Δ² y')
    plt.grid(True)
    plt.show()
