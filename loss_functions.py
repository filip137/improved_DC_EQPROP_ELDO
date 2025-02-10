#######################################################################
#                               imports                               #
#######################################################################

import numpy as np

#######################################################################
#                    methods for calculating loss                     #
#######################################################################

class MSE:
    def __init__(self, boundary):
        self.boundary = boundary

    def __call__(self, output_node_voltages, target=None, beta=None, mode='train'):
        """
        Calculate losses and gradient currents according to the
        Mean Squared Error (MSE)
        """
        """
        Calculate losses and gradient currents according to the
        Mean Squared Error (MSE).
        Handles both dict and list input formats for output_node_voltages.
        """
        # Check if output_node_voltages is a dictionary
        if isinstance(output_node_voltages, dict):
            # Convert dict values to a NumPy array
            output_node_voltages_values = np.array(list(output_node_voltages.values()))
            num_output_nodes = len(output_node_voltages)
        elif isinstance(output_node_voltages, list):
            # Convert the list to a NumPy array directly
            output_node_voltages_values = np.array(output_node_voltages)
            num_output_nodes = output_node_voltages_values.shape[1]
        else:
            raise ValueError("output_node_voltages must be either a dictionary or a list.")
        
        #Number of outputs
        # Reshape into a (rows, 2) array
        output_nodes_voltages = output_node_voltages_values.reshape(-1, 2)
        prediction = output_nodes_voltages[:, 0] - output_nodes_voltages[:, 1] # array of predictions

        if mode == 'train':

            losses = np.zeros(shape=(int(num_output_nodes/2), ))
            currents = np.zeros(shape=(int(num_output_nodes/2), 2))

            # MSE calculatuon
            diff = prediction - target
            losses = 0.5 * np.power(diff, 2)

            # loss current calculation
            #beta = np.random.choice([-1, 1]) * beta
            currents[:,0] = -beta * diff
            currents[:,1] = beta * diff
            return losses, currents

        return prediction.reshape(-1, int(num_output_nodes/2))

    def verify_result(self, target, prediction):
        # Convert predictions to class indices
        prediction_indices = np.argmax(prediction, axis=1)  # assuming prediction is shape (N, num_classes)
        
        # Convert target from one-hot encoded to class indices
        target_indices = np.argmax(target, axis=1)  # assuming target is shape (N, num_classes)
        
        # Compare prediction_indices and target_indices
        correct_predictions = (prediction_indices == target_indices).astype(int)  # Result is 0 for false, 1 for true
        
        return correct_predictions
    
    
    def binary_prediction(self, prediction):
        prediction_indices = np.argmax(prediction, axis=1)
        return (prediction_indices).astype(int)


class BCE:
    def __call__(self, output_node_voltages, target=None, beta=None, mode='train'):
        """
        Calculate losses and gradient currents according to the
        Binary Cross Entropy Loss (BCE)
        """
        prediction = output_node_voltages[:,0] - output_node_voltages[:,1]
        prob = 1 / (1 + np.exp(-prediction)) # sigmoid function

        if mode == 'train':
            eps = 0.0000001
            losses = np.zeros(shape=(1, ))
            currents = np.zeros(shape=(1, 2))

            losses = - ((target * np.log10(prob + eps) + (1 - target) * np.log(1 - prob + eps)))
            diff = prob - target

            currents[:,0] = -beta * diff
            currents[:,1] = beta * diff
            return losses, currents

        return prob

    def verify_result(self, target, prediction):
        c = np.product(np.equal(target, np.round(prediction, 0)), axis=1)
        return c

class CrossEntropyLoss:
    def __call__(self, output_node_voltages, target=None, beta=None, mode='train'):
        """
        Calculate losses and gradient currents according to the
        Cross Entropy Loss
        """
        prediction = output_node_voltages[:,0] - output_node_voltages[:,1]
        prob = np.exp(prediction) / np.sum(np.exp(prediction))

        if mode == 'train':
            eps = 0.0000001
            output_nodes = output_node_voltages.shape[0]
            losses = np.zeros(shape=(output_nodes, ))
            currents = np.zeros(shape=(output_nodes, 2))

            # cross-entropy loss calculation
            losses = - target * np.log(prob + eps)
            diff = prob - target

            # loss current calculation
            currents[:,0] = -beta * diff
            currents[:,1] = beta * diff
            return losses, currents

        return prob