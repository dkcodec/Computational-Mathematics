# Forward Difference Table Construction and Verification of Constant Third-Order Differences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def forward_difference_table(x, y):
    """
    Constructs the forward difference table for given x and y data points.

    Parameters:
    x (list or np.array): The x-values.
    y (list or np.array): The y-values.

    Returns:
    pd.DataFrame: A pandas DataFrame representing the forward difference table.
    """
    n = len(y)
    # Initialize a list of lists to store differences
    diff_table = [y.copy()]
    
    # Compute forward differences up to the (n-1)th order
    for order in range(1, n):
        previous_diff = diff_table[-1]
        current_diff = []
        for i in range(len(previous_diff) - 1):
            delta = previous_diff[i+1] - previous_diff[i]
            current_diff.append(delta)
        diff_table.append(current_diff)
    
    # Determine the maximum number of columns needed
    max_cols = n
    # Initialize a DataFrame with NaN values
    df = pd.DataFrame(np.nan, index=range(n), columns=range(max_cols))
    
    # Fill in the DataFrame with x and differences
    df.iloc[:,0] = x  # First column for x-values
    df.iloc[0,1] = y[0]  # First y-value
    for i in range(1, n):
        df.iloc[i,0] = x[i]
        for j in range(1, i+1):
            df.iloc[i,j] = diff_table[j][i - j]
    
    # Rename columns for clarity
    column_names = ['x']
    for i in range(1, max_cols):
        column_names.append(f'Δ^{i}y')
    df.columns = column_names
    
    return df

# Data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 7, 13, 21])

# Construct the forward difference table
diff_table_df = forward_difference_table(x, y)

# Display the forward difference table
print("Forward Difference Table:")
print(diff_table_df.to_string(index=False))

# Verify if the third-order differences are constant
# Extract the third-order differences
third_order_diffs = diff_table_df['Δ^3y'].dropna().values

# Check if all third-order differences are equal
if len(third_order_diffs) > 0:
    is_constant = np.allclose(third_order_diffs, third_order_diffs[0])
    print("\nThird-order differences:", third_order_diffs)
    print("Are the third-order differences constant?", is_constant)
else:
    print("\nNot enough data to compute third-order differences.")

# Plot the forward differences (optional)
# Plotting third-order differences to visualize constancy
if len(third_order_diffs) > 0:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(third_order_diffs)+1), third_order_diffs, marker='o', linestyle='-', color='green')
    plt.title('Third-Order Forward Differences')
    plt.xlabel('Order Index')
    plt.ylabel('Δ^3 y')
    plt.grid(True)
    plt.show()
