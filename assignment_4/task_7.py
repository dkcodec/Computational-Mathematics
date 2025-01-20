# Forward Difference Table Construction and Verification of Constant Fourth-Order Differences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def forward_difference_table(x, y, max_order=4):
    """
    Constructs the forward difference table for given x and y data points.

    Parameters:
    x (list or np.array): The x-values.
    y (list or np.array): The y-values.
    max_order (int): The maximum order of differences to compute.

    Returns:
    pd.DataFrame: A pandas DataFrame representing the forward difference table.
    """
    n = len(y)
    # Initialize a list to store each order of differences
    diff_table = [y.copy()]
    
    # Compute forward differences up to the specified max_order
    for order in range(1, max_order + 1):
        previous_diff = diff_table[-1]
        current_diff = []
        for i in range(len(previous_diff) - 1):
            delta = previous_diff[i+1] - previous_diff[i]
            current_diff.append(delta)
        # Append the current order differences to the table
        diff_table.append(current_diff)
    
    # Create column names based on the order of differences
    column_names = ['x']
    for order in range(1, max_order + 1):
        column_names.append(f'Δ^{order}y')
    
    # Initialize the DataFrame with NaN values
    df = pd.DataFrame(np.nan, index=range(n), columns=column_names)
    
    # Fill in the x-values
    df['x'] = x
    
    # Populate the DataFrame with computed differences
    for order in range(1, max_order + 1):
        for i in range(len(diff_table[order])):
            df.at[i, f'Δ^{order}y'] = diff_table[order][i]
    
    return df

# Data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 8, 27, 64, 125])

# Construct the forward difference table up to the fourth order
diff_table_df = forward_difference_table(x, y, max_order=4)

# Display the forward difference table
print("Forward Difference Table:")
print(diff_table_df.to_string(index=False))

# Verify if the fourth-order differences are constant
# Extract the fourth-order differences
fourth_order_diffs = diff_table_df['Δ^4y'].dropna().values

# Check if all fourth-order differences are equal (constant)
if len(fourth_order_diffs) > 0:
    is_constant = np.allclose(fourth_order_diffs, fourth_order_diffs[0])
    print("\nFourth-order differences:", fourth_order_diffs)
    print("Are the fourth-order differences constant?", is_constant)
else:
    print("\nNot enough data to compute fourth-order differences.")

# Plot the fourth-order differences (optional)
if len(fourth_order_diffs) > 0:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(fourth_order_diffs)+1), fourth_order_diffs, marker='o', linestyle='-', color='orange')
    plt.title('Fourth-Order Forward Differences')
    plt.xlabel('Index')
    plt.ylabel('Δ⁴ y')
    plt.grid(True)
    plt.show()
