###PLOTTING
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def output_plot(data):
    """
    Plots six columns from a list of dict_values, and their differences between columns 3 and 4, and columns 5 and 6.

    Parameters:
        data (list): A list of dict_values objects containing six numerical values each.
    """
    # Convert dict_values to a list of lists
    converted_data = [list(values) for values in data]

    # Extract columns
    columns = [[row[i] for row in converted_data] for i in range(6)]

    # Calculate the differences
    difference0 = [x - y for x, y in zip(columns[0], columns[1])]  # Difference between columns 3 and 4

    difference1 = [x - y for x, y in zip(columns[2], columns[3])]  # Difference between columns 3 and 4
    difference2 = [x - y for x, y in zip(columns[4], columns[5])]  # Difference between columns 5 and 6

    # Create plots for each column
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    titles = ['Column 1', 'Column 2', 'Column 3', 'Column 4', 'Column 5', 'Column 6']
    for i, (column, color, title) in enumerate(zip(columns, colors, titles)):
        plt.figure()
        plt.plot(column, label=title, marker='o', color=color)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.show()




    # Plot the differences
    plt.figure()
    plt.plot(difference0, label="Difference (Column 1 - Column 2)", marker='o', color='black')
    plt.xlabel("Index")
    plt.ylabel("Difference")
    plt.title("Difference Between Column 1 and Column 2")
    plt.legend()
    plt.show()

    # Plot the differences
    plt.figure()
    plt.plot(difference1, label="Difference (Column 3 - Column 4)", marker='o', color='black')
    plt.xlabel("Index")
    plt.ylabel("Difference")
    plt.title("Difference Between Column 3 and Column 4")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(difference2, label="Difference (Column 5 - Column 6)", marker='o', color='grey')
    plt.xlabel("Index")
    plt.ylabel("Difference")
    plt.title("Difference Between Column 5 and Column 6")
    plt.legend()
    plt.show()

# Example usage
# data = [{values of 6 columns per dict_values}, ...]
# output_plot(data)


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


def plot_weight_matrix_evolution_separate(
    weight_matrices,
    interval,
    title,
    num_random_weights=4,
    random_seed=None
):
    """
    Plots the evolution of weights in the matrices over iterations,
    with each weight in a separate graph, but only for a random subset.

    Parameters:
        weight_matrices (list of 2D arrays): List of weight matrices (one for each iteration).
        interval (int): Interval of iterations to plot (default is 1, plots all iterations).
        title (str): Title prefix for each plot.
        num_random_weights (int): Number of random weights to plot.
        random_seed (int or None): If provided, sets a random seed for reproducibility.
    """

    # Optional: set a random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert the list of matrices into a 3D NumPy array (iterations, rows, cols)
    weight_matrices_array = np.array(weight_matrices)
    num_iterations, num_rows, num_cols = weight_matrices_array.shape

    # Flatten each weight matrix to track individual weights
    flattened_weights = weight_matrices_array.reshape(num_iterations, -1)
    num_weights = flattened_weights.shape[1]

    # Downsample iterations based on the interval
    x = np.arange(0, num_iterations, interval)
    sampled_weights = flattened_weights[x]

    # --- NEW: Select a random subset of weights ---
    # We only choose num_random_weights distinct indices
    # If num_random_weights > num_weights, we'll plot all weights
    num_random_weights = min(num_random_weights, num_weights)
    random_weight_indices = np.random.choice(num_weights, size=num_random_weights, replace=False)
    
    # Create a separate graph for each randomly chosen weight
    for i, weight_idx in enumerate(random_weight_indices, start=1):
        plt.figure(figsize=(8, 6))
        plt.plot(x, sampled_weights[:, weight_idx], label=f'Weight {weight_idx+1}', alpha=0.8)
        plt.yscale('log')  # Log scale for the y-axis if needed
        plt.title(f"{title} - Random Weight {i} (Index {weight_idx+1})")
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



def plot_loss(loss_lists):
    """
    Plots the summed training loss over iterations when given multiple losses per iteration.
    
    Parameters:
        loss_lists (list of lists or array-like): A list where each element is a list containing multiple loss values at each iteration.
    """
    # Summing up the losses for each iteration
    summed_losses = [sum(losses) for losses in loss_lists]

    plt.figure(figsize=(8, 5))
    plt.plot(summed_losses, label='Summed Training Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Summed Training Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()