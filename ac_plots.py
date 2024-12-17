###PLOTTING

import matplotlib.pyplot as plt

def output_plot(data):
    """
    Plots the first column, second column, and their difference from a list of dict_values.

    Parameters:
        data (list): A list of dict_values objects containing two numerical values each.
    """
    # Convert dict_values to a list of lists
    converted_data = [list(values) for values in data]

    # Extract columns
    column1 = [row[0] for row in converted_data]  # First column
    column2 = [row[1] for row in converted_data]  # Second column

    # Calculate the difference
    difference = [x - y for x, y in zip(column1, column2)]

    # Plot the first column
    plt.figure()
    plt.plot(column1, label="Column 1", marker='o')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Column 1")
    plt.legend()
    plt.show()

    # Plot the second column
    plt.figure()
    plt.plot(column2, label="Column 2", marker='o', color='orange')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Column 2")
    plt.legend()
    plt.show()

    # Plot the difference
    plt.figure()
    plt.plot(difference, label="Difference (Column 1 - Column 2)", marker='o', color='green')
    plt.xlabel("Index")
    plt.ylabel("Difference")
    plt.title("Difference Between Column 1 and Column 2")
    plt.legend()
    plt.show()