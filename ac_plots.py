###PLOTTING
import numpy as np
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

def plot_weight_matrix_evolution_lines(weight_matrices, interval, title):
    """
    Plots the evolution of all weights in the matrices over iterations.

    Parameters:
        weight_matrices (list of 2D arrays): List of weight matrices (one for each iteration).
        interval (int): Interval of iterations to plot (default is 1, plots all iterations).
    """
    # Convert the list of matrices into a 3D NumPy array (iterations, rows, cols)
    weight_matrices_array = np.array(weight_matrices)
    num_iterations, num_rows, num_cols = weight_matrices_array.shape

    # Flatten each weight matrix to track individual weights
    flattened_weights = weight_matrices_array.reshape(num_iterations, -1)

    # Downsample iterations based on the interval
    x = np.arange(0, num_iterations, interval)
    sampled_weights = flattened_weights[x]

    # Create a line plot for each weight
    plt.figure(figsize=(12, 8))
    for i in range(sampled_weights.shape[1]):  # Number of weights
        plt.plot(x, sampled_weights[:, i], label=f'Weight {i+1}', alpha=0.7)

    plt.yscale('log')
    # Add plot labels and legend
    plt.title(f"{title}")
    plt.xlabel("Iteration")
    plt.ylabel("Weight Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()  

def plot_free_and_nudged(output_list, output_list_nudge, output_nodes, beta, gamma):
    """
    Plots free and nudged results for each output node over iterations.

    Parameters:
        output_list (list of lists): List of lists containing free results (e.g., [[val1, val2], ...]).
        output_list_nudge (list of lists): List of lists containing nudged results (e.g., [[val1, val2], ...]).
        output_nodes (list): List of output node indices to plot (e.g., [0, 1]).
        beta (float): Beta parameter for annotation.
        gamma (float): Gamma parameter for annotation.
    """
    # Validate input
    if not isinstance(output_list, list) or not isinstance(output_list_nudge, list):
        raise ValueError("output_list and output_list_nudge must be lists")
    if not output_list or not output_list_nudge:
        raise ValueError("Input data cannot be empty")
    if len(output_list) != len(output_list_nudge):
        raise ValueError("Both lists must have the same length")
    
    # Ensure each inner list has the correct length
    num_nodes = len(output_list[0])
    if not all(len(item) == num_nodes for item in output_list):
        raise ValueError("Each element of output_list must have the same length")
    if not all(len(item) == num_nodes for item in output_list_nudge):
        raise ValueError("Each element of output_list_nudge must have the same length")

    # Number of iterations
    n_of_iter = len(output_list)
    x = np.linspace(1, n_of_iter, n_of_iter)

    # Extract data and plot for each output node
    for node in output_nodes:
        # Extract the data for the current node
        free_results = np.array([result[node] for result in output_list])
        nudged_results = np.array([result[node] for result in output_list_nudge])

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(x, free_results, label=f'Free Results (Node {node})', marker='o')
        plt.plot(x, nudged_results, label=f'Nudged Results (Node {node})', marker='x')
        plt.plot(x, free_results - nudged_results, label=f'Difference (Node {node})', linestyle='--')
        
        # Adding titles and labels
        plt.title(f"Results over Iterations for Node {node} (beta={beta}, gamma={gamma})")
        plt.xlabel('Iteration')
        plt.ylabel('Measured Value')

        # Add legend
        plt.legend(loc='best')

        # Enable grid
        plt.grid(True)

        # Show plot
        plt.show()
