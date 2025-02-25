###PLOTTING
import numpy as np
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def output_plot(data):
    """
    Plots six columns from a list of dict_values, and their differences between columns 3 and 4, and columns 5 and 6.

    Parameters:
        data (list): A list of dict_values objects containing four or six numerical values each.
    """
    # Convert dict_values to a list of lists
    converted_data = [list(values) for values in data]

    # Extract columns
    columns = [[row[i] for row in converted_data] for i in range(4)]

    # Calculate the differences
    difference0 = [x - y for x, y in zip(columns[0], columns[1])]  # Difference between columns 3 and 4

    difference1 = [x - y for x, y in zip(columns[2], columns[3])]  # Difference between columns 3 and 4
    #difference2 = [x - y for x, y in zip(columns[4], columns[5])]  # Difference between columns 5 and 6

    # Create plots for each column
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    titles = ['Column 1', 'Column 2', 'Column 3', 'Column 4', 'Column 5', 'Column 6']
    # for i, (column, color, title) in enumerate(zip(columns, colors, titles)):
    #     plt.figure()
    #     plt.plot(column, label=title, marker='o', color=color)
    #     plt.xlabel("Index")
    #     plt.ylabel("Value")
    #     plt.title(title)
    #     plt.legend()
    #     plt.show()




    # # Plot the differences
    # plt.figure()
    # plt.plot(difference0, label="Difference (Column 1 - Column 2)", marker='o', color='black')
    # plt.xlabel("Index")
    # plt.ylabel("Difference")
    # plt.title("Difference Between Column 1 and Column 2")
    # plt.legend()
    # plt.show()

    # # Plot the differences
    # plt.figure()
    # plt.plot(difference1, label="Difference (Column 3 - Column 4)", marker='o', color='black')
    # plt.xlabel("Index")
    # plt.ylabel("Difference")
    # plt.title("Difference Between Column 3 and Column 4")
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(difference2, label="Difference (Column 5 - Column 6)", marker='o', color='grey')
    # plt.xlabel("Index")
    # plt.ylabel("Difference")
    # plt.title("Difference Between Column 5 and Column 6")
    # plt.legend()
    # plt.show()
    return difference0, difference1
# Example usage
# data = [{values of 6 columns per dict_values}, ...]
# output_plot(data)

def moving_average_np(data, window_size):
    data = np.array(data)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > len(data):
        raise ValueError("window_size cannot be larger than the length of the data")
    
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_moving_averages(diff1_list, diff2_list, window_size=10):
    """
    Calculates and plots the moving averages of two lists.

    Parameters:
        diff1_list (list): First list of numerical data.
        diff2_list (list): Second list of numerical data.
        window_size (int): Window size for the moving average.
    """
    # Calculate moving averages
    ma_diff1 = moving_average_np(diff1_list, window_size)
    ma_diff2 = moving_average_np(diff2_list, window_size)
    
    # Create x-axis values (indices)
    x1 = np.arange(len(ma_diff1))
    x2 = np.arange(len(ma_diff2))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x1, ma_diff1, label='Moving Average of diff1_list')
    plt.plot(x2, ma_diff2, label='Moving Average of diff2_list')
    
    plt.xlabel('Index')
    plt.ylabel('Moving Average')
    plt.title(f'Moving Averages (window size = {window_size})')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(accuracy_list):
    """
    Plots accuracy values from the given list.
    
    Args:
        accuracy_list (list or array): List of accuracy values.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(accuracy_list, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.grid(True)
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



def plot_weight_matrix_evolution_heatmaps(weight_matrices, interval=1, title='Weight Matrix Evolution'):
    selected_matrices = weight_matrices[::interval]
    
    for i, W in enumerate(selected_matrices):
        plt.figure(figsize=(5, 4))
        im = plt.imshow(W, aspect='auto', cmap='viridis')
        plt.title(f"Epoch {i * interval} - {title}")
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()

def plot_weight_histogram(weight_matrix, bins=50, title='Weight Matrix Histogram'):
    """
    Displays a histogram of the weight matrix.
    
    Parameters:
    -----------
    weight_matrix : np.ndarray
        A 2D (or higher-dimensional) NumPy array containing weight values.
    bins : int, optional
        Number of bins for the histogram (default is 50).
    title : str, optional
        The title of the histogram plot.
    """
    # Flatten the weight matrix into a 1D array
    weights = weight_matrix.flatten()
    
    # Create the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(weights, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()




def plot_free_and_nudged(output_list, output_list_nudge, output_nodes, beta, gamma, epoch):
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
        plt.title(f"Results over Iterations for Node {node} (beta={beta}, gamma={gamma}, epoch {epoch})")
        plt.xlabel('Iteration')
        plt.ylabel('Measured Value')

        # Add legend
        plt.legend(loc='best')

        # Enable grid
        plt.grid(True)

        # Show plot
        plt.show()


def plot_average_loss(loss_list, title='Average Loss per Epoch', xlabel='Epoch', ylabel='Loss',
                      marker='o', linestyle='-', color='b', grid=True):
    """
    Computes the average loss for each epoch from a list of arrays and plots it.
    
    Parameters:
        loss_list (list of np.array): A list where each element is a NumPy array of loss values for an epoch.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        marker (str): Marker style for the plot (default is 'o').
        linestyle (str): Line style for the plot (default is '-').
        color (str): Color for the plot (default is blue, 'b').
        grid (bool): If True, displays a grid on the plot.
    """
    # Convert the list to a NumPy array and compute the mean loss per epoch
    loss_array = np.array(loss_list)
    avg_loss = np.mean(loss_array, axis=1)
    
    # Create an array for the epoch numbers
    epochs = np.arange(1, avg_loss.shape[0] + 1)
    
    # Plot the average loss per epoch
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_loss, marker=marker, linestyle=linestyle, color=color, label='Average Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid(True)
    plt.legend()
    plt.show()


def compute_relative_changes(weight_matrices, epsilon=1e-10):
    """
    Computes the relative change of the weights from one epoch to the next.
    
    Parameters
    ----------
    weight_matrices : list of np.ndarray
        A list where each element is a weight matrix at a given epoch.
    epsilon : float, optional
        A small constant to avoid division by zero (default: 1e-8).
        
    Returns
    -------
    rel_changes : list of np.ndarray
        A list of relative change matrices. Each element corresponds to the 
        relative change computed as:
            (W[t] - W[t-1]) / (abs(W[t-1]) + epsilon)
        for t = 1, 2, ..., len(weight_matrices)-1.
    """
    rel_changes = []
    for t in range(1, len(weight_matrices)):
        prev = weight_matrices[t-1]
        curr = weight_matrices[t]
        change = (curr - prev) / (np.abs(prev) + epsilon)
        rel_changes.append(change)
    return rel_changes

def plot_average_relative_change(rel_changes, title="Average Relative Weight Change per Epoch"):
    """
    Plots a continuous line graph of the average (absolute) relative change per epoch.
    
    Parameters
    ----------
    rel_changes : list of np.ndarray
        List of relative change matrices (output of compute_relative_changes).
    title : str, optional
        Title for the plot.
    """
    # Compute the average absolute relative change for each epoch.
    avg_changes = [np.mean(np.abs(change)) for change in rel_changes]
    epochs = np.arange(1, len(avg_changes) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, avg_changes, marker='o', linestyle='-', color='blue')
    plt.xlabel('Epoch (relative change computed from previous epoch)')
    plt.ylabel('Average Absolute Relative Change')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_weight_evolution(weight_matrices, title='Weight Evolution Over Epochs'):
    """
    Plots the evolution of individual weights and the mean weight value across epochs.
    
    Parameters
    ----------
    weight_matrices : list of np.ndarray
        List where each element is a weight matrix from a given epoch.
    title : str, optional
        Title for the overall figure (default is 'Weight Evolution Over Epochs').
    """
    epochs = len(weight_matrices)
    # Flatten each weight matrix into a 1D array.
    evolution = np.array([W.flatten() for W in weight_matrices])
    n_weights = evolution.shape[1]
    epoch_range = np.arange(epochs)
    
    # Create a figure with two subplots:
    # Left: individual weight evolutions.
    # Right: mean weight evolution.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot each weight's evolution as a line (with slight transparency)
    for i in range(n_weights):
        axs[0].plot(epoch_range, evolution[:, i], alpha=0.5)
    axs[0].set_title("Individual Weight Evolution")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Weight Value")
    axs[0].grid(True)
    
    # Compute and plot the mean weight at each epoch.
    mean_weights = np.mean(evolution, axis=1)
    axs[1].plot(epoch_range, mean_weights, marker='o', color='red')
    axs[1].set_title("Mean Weight Evolution")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Mean Weight")
    axs[1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_loss(loss_lists, epoch):
    """
    Plots the training loss over iterations. If each loss entry in loss_lists
    contains only one value, it plots those values directly. Otherwise, it sums
    the losses for each iteration before plotting.

    Parameters:
        loss_lists (list of lists or array-like): A list where each element is a list 
            containing one or more loss values at each iteration.
        epoch (int): The current epoch number for title labeling.
    """
    # Check if every element in loss_lists has exactly one loss value
    if all(len(loss) == 1 for loss in loss_lists):
        plot_values = [loss[0] for loss in loss_lists]
    else:
        plot_values = [sum(loss) for loss in loss_lists]

    plt.figure(figsize=(8, 5))
    plt.plot(plot_values, label='Training Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Iterations for Epoch {epoch}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_cosine_similarity(cosine_list, title="Cosine similarity"):

    # Extract ratios from the list of dicts
    #weight_ratios = [d['weight_reduction_ratio'] for d in ratio_dicts]
    
    iterations = range(1, len(cosine_list) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cosine_list, label="Cosine similarity", color="blue", marker='o')
    #plt.plot(iterations, weight_ratios, label="Weight Reduction Ratio", color="red", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Cosine")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()   
    
def plot_reduction_ratios_from_dicts(ratio_dicts, title="Reduction Ratios Over Iterations"):
    """
    Plots the gradient and weight reduction ratios over iterations from a list of dictionaries.

    Parameters:
        ratio_dicts (list of dict): Each dict should contain:
            - 'gradient_reduction_ratio': The gradient reduction ratio for that iteration.
            - 'weight_reduction_ratio': The weight reduction ratio for that iteration.
        title (str): Title of the plot.
    """
    # Extract ratios from the list of dicts
    grad_ratios = [d['gradient_reduction_ratio'] for d in ratio_dicts]
    #weight_ratios = [d['weight_reduction_ratio'] for d in ratio_dicts]
    
    iterations = range(1, len(ratio_dicts) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, grad_ratios, label="Gradient Reduction Ratio", color="blue", marker='o')
    #plt.plot(iterations, weight_ratios, label="Weight Reduction Ratio", color="red", marker='s')
    plt.xlabel("Iteration")
    plt.ylabel("Reduction Ratio")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()